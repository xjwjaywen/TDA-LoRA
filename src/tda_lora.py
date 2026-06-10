"""
TDA-LoRA: Timestep-Domain Adaptive Low-Rank Adaptation.

Uses PEFT's standard LoRA as backbone, adds three adaptive components:
  1. Timestep-aware loss weighting (sample fewer high-noise timesteps)
  2. Domain-aware rank adjustment (CLIP-based gap -> per-layer rank)
  3. Gradient-guided layer importance (warmup -> rank_pattern)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model


def create_tda_lora(
    unet: nn.Module,
    target_modules: List[str],
    base_rank: int = 8,
    alpha: float = 16.0,
    rank_pattern: Optional[Dict[str, int]] = None,
    alpha_pattern: Optional[Dict[str, float]] = None,
):
    """Create LoRA-adapted UNet with optional per-layer rank pattern."""
    config = LoraConfig(
        r=base_rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        rank_pattern=rank_pattern or {},
        alpha_pattern=alpha_pattern or {},
    )
    unet = get_peft_model(unet, config)
    unet.print_trainable_parameters()
    return unet


class TimestepSampler:
    """Timestep-aware sampling distribution for training.

    Key insight from T-LoRA: high-noise timesteps are more prone to overfitting.
    We sample fewer high-noise timesteps and more mid/low-noise timesteps.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        strategy: str = "tda",  # "uniform", "tda", "mid_focus"
        domain_gap: float = 0.0,
    ):
        self.num_timesteps = num_timesteps
        self.strategy = strategy
        self.domain_gap = domain_gap
        self.weights = self._build_weights()

    def _build_weights(self) -> torch.Tensor:
        t = torch.linspace(0, 1, self.num_timesteps)

        if self.strategy == "uniform":
            w = torch.ones(self.num_timesteps)

        elif self.strategy == "tda":
            # Base: reduce high-noise timestep probability
            # Higher domain gap -> more conservative (less high-noise sampling)
            high_noise_scale = max(0.2, 1.0 - self.domain_gap * 0.8)
            w = torch.where(t > 0.66, high_noise_scale, torch.ones_like(t))
            w = torch.where(t < 0.33, torch.ones_like(t) * 1.2, w)  # boost low-noise
            # Mid-range gets slight boost
            mid_mask = (t >= 0.33) & (t <= 0.66)
            w[mid_mask] = 1.5

        elif self.strategy == "mid_focus":
            # Only mid-range boost, no domain awareness
            w = torch.ones(self.num_timesteps)
            mid_mask = (t >= 0.25) & (t <= 0.75)
            w[mid_mask] = 2.0

        else:
            w = torch.ones(self.num_timesteps)

        w = w / w.sum()
        return w

    def sample(self, batch_size: int, device: str = "cuda") -> torch.Tensor:
        indices = torch.multinomial(self.weights, batch_size, replacement=True)
        return indices.to(device)


def compute_domain_gap(
    target_image_paths: list,
    pretrained_model_id: str,
    device: str = "cuda",
) -> float:
    """Estimate domain gap using CLIP features."""
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    images = [Image.open(p).convert("RGB") for p in target_image_paths]
    img_tensors = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        img_features = model.encode_image(img_tensors)
        img_features = F.normalize(img_features, dim=-1)
        target_centroid = img_features.mean(dim=0)

    ref_prompts = [
        "a photo", "a natural image", "a picture of an object",
        "a photograph", "a realistic image",
    ]
    tokens = tokenizer(ref_prompts).to(device)
    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)
        ref_centroid = text_features.mean(dim=0)

    gap = 1.0 - F.cosine_similarity(
        target_centroid.unsqueeze(0), ref_centroid.unsqueeze(0)
    ).item()

    del model
    torch.cuda.empty_cache()

    gap = max(0.0, min(1.0, gap / 0.8))
    print(f"Domain gap score: {gap:.3f}")
    return gap


def compute_layer_importance(
    unet,
    dataloader,
    noise_scheduler,
    text_encoder,
    vae,
    tokenizer,
    warmup_steps: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """Gradient-guided layer importance via short warmup training."""
    print(f"Computing layer importance ({warmup_steps} warmup steps)...")

    grad_norms = {}
    for name, param in unet.named_parameters():
        if "lora_A" in name and param.requires_grad:
            grad_norms[name.replace(".lora_A.default.weight", "")] = 0.0

    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad], lr=1e-4
    )

    unet.train()
    data_iter = iter(dataloader)

    for step in range(warmup_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, dtype=torch.float16)
        prompt = batch["prompt"][0]

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            tokens = tokenizer(
                [prompt] * latents.shape[0], padding="max_length",
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)
            encoder_hidden_states = text_encoder(tokens)[0]

        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()

        for name, param in unet.named_parameters():
            if "lora_A" in name and param.requires_grad and param.grad is not None:
                key = name.replace(".lora_A.default.weight", "")
                if key in grad_norms:
                    grad_norms[key] += param.grad.norm().item()

        optimizer.step()
        optimizer.zero_grad()

    # Normalize
    max_norm = max(grad_norms.values()) if grad_norms else 1.0
    importance = {k: v / max_norm for k, v in grad_norms.items()}

    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v:.3f}")

    return importance


def build_rank_pattern(
    importance: Dict[str, float],
    base_rank: int,
    domain_gap: float,
    domain_scale_factor: float = 0.3,
    top_k_ratio: float = 0.5,
) -> Dict[str, int]:
    """Convert layer importance + domain gap into per-layer rank pattern."""
    # Domain-adjusted base rank
    effective_base = max(2, int(base_rank * (1 - domain_scale_factor * domain_gap)))
    print(f"Domain gap {domain_gap:.3f} -> effective base rank: {effective_base}")

    sorted_layers = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_k = int(len(sorted_layers) * top_k_ratio)

    rank_pattern = {}
    alpha_pattern = {}
    for i, (name, imp) in enumerate(sorted_layers):
        if i < top_k:
            rank = min(effective_base * 2, 16)  # important layers get 2x rank
        else:
            rank = max(2, effective_base // 2)   # less important layers get 0.5x rank
        rank_pattern[name] = rank
        alpha_pattern[name] = float(rank * 2)  # keep alpha/rank ratio = 2

    print(f"Rank pattern: {len([r for r in rank_pattern.values() if r > effective_base])} layers boosted, "
          f"{len([r for r in rank_pattern.values() if r <= effective_base])} layers reduced")
    return rank_pattern, alpha_pattern
