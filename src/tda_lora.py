"""
AdaSNR-LoRA: Domain-Adaptive SNR Loss Reweighting for Few-Shot LoRA.

Built on proven techniques:
  1. Min-SNR loss weighting (ICCV 2023) - reweight loss per timestep
  2. Domain-adaptive gamma - adjust SNR gamma based on domain gap
  3. Gradient-guided layer rank - important layers get higher rank

Novel contribution: domain-adaptive SNR reweighting where gamma is
automatically determined by the target domain's characteristics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from peft import LoraConfig, get_peft_model


def create_lora(
    unet: nn.Module,
    target_modules: List[str],
    rank: int = 16,
    alpha: float = 16.0,
    rank_pattern: Optional[Dict[str, int]] = None,
    alpha_pattern: Optional[Dict[str, float]] = None,
):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        rank_pattern=rank_pattern or {},
        alpha_pattern=alpha_pattern or {},
    )
    unet = get_peft_model(unet, config)
    unet.print_trainable_parameters()
    return unet


# =============================================
# Core Innovation: Domain-Adaptive Min-SNR
# =============================================

def compute_snr(noise_scheduler, timesteps):
    """Compute signal-to-noise ratio for given timesteps."""
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
    sqrt_alpha = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alpha = (1 - alphas_cumprod[timesteps]) ** 0.5
    snr = (sqrt_alpha / sqrt_one_minus_alpha) ** 2
    return snr


def compute_min_snr_weights(noise_scheduler, timesteps, gamma=5.0):
    """Min-SNR-gamma loss weighting (ICCV 2023)."""
    snr = compute_snr(noise_scheduler, timesteps)
    snr_clipped = torch.clamp(snr, max=gamma)
    weights = snr_clipped / snr
    return weights


def compute_adaptive_snr_weights(noise_scheduler, timesteps, gamma, domain_gap, domain_boost=1.0):
    """
    Domain-Adaptive SNR weighting (our contribution).

    For high domain gap: boost mid-range timestep weights (where domain
    structure is learned) and reduce high-noise weights more aggressively.
    """
    snr = compute_snr(noise_scheduler, timesteps)

    # Adaptive gamma: higher domain gap -> lower gamma -> less high-noise influence
    adaptive_gamma = gamma * (1.0 - 0.4 * domain_gap)
    adaptive_gamma = max(1.0, adaptive_gamma)

    snr_clipped = torch.clamp(snr, max=adaptive_gamma)
    weights = snr_clipped / snr

    # Domain boost: extra weight for mid-range timesteps when domain gap is large
    if domain_boost > 0 and domain_gap > 0.1:
        alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
        t_normalized = alphas_cumprod[timesteps]
        # Gaussian centered at mid-range (alpha_cumprod ~= 0.5)
        mid_boost = torch.exp(-((t_normalized - 0.5) ** 2) / (2 * 0.15 ** 2))
        boost_factor = 1.0 + domain_boost * domain_gap * mid_boost
        weights = weights * boost_factor

    return weights


# =============================================
# Domain Gap Estimation
# =============================================

def compute_domain_gap(target_image_paths: list, device: str = "cuda") -> float:
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

    ref_prompts = ["a photo", "a natural image", "a picture of an object",
                   "a photograph", "a realistic image"]
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


# =============================================
# Layer Importance (for rank pattern)
# =============================================

def compute_layer_importance(
    unet, dataloader, noise_scheduler, text_encoder, vae, tokenizer,
    warmup_steps: int = 30, device: str = "cuda",
) -> Dict[str, float]:
    """Quick gradient-guided layer importance estimation."""
    print(f"Computing layer importance ({warmup_steps} steps)...")

    grad_norms = {}
    for name, param in unet.named_parameters():
        if "lora_A" in name and param.requires_grad:
            key = name.replace(".lora_A.default.weight", "")
            grad_norms[key] = 0.0

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
            enc_hidden = text_encoder(tokens)[0]

        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                          (latents.shape[0],), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, t)

        pred = unet(noisy, t, enc_hidden).sample
        loss = F.mse_loss(pred.float(), noise.float())
        loss.backward()

        for name, param in unet.named_parameters():
            if "lora_A" in name and param.requires_grad and param.grad is not None:
                key = name.replace(".lora_A.default.weight", "")
                if key in grad_norms:
                    grad_norms[key] += param.grad.norm().item()

        optimizer.step()
        optimizer.zero_grad()

    max_norm = max(grad_norms.values()) if grad_norms else 1.0
    importance = {k: v / max_norm for k, v in grad_norms.items()}
    return importance


def build_rank_pattern(
    importance: Dict[str, float],
    base_rank: int,
    top_k_ratio: float = 0.5,
) -> tuple:
    """Assign higher rank to important layers, lower to unimportant ones."""
    sorted_layers = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top_k = max(1, int(len(sorted_layers) * top_k_ratio))

    rank_pattern = {}
    alpha_pattern = {}
    boosted = 0
    for i, (name, imp) in enumerate(sorted_layers):
        if i < top_k:
            r = min(base_rank + 8, 32)  # boost by 8, cap at 32
            boosted += 1
        else:
            r = max(4, base_rank - 4)  # reduce by 4, floor at 4
        rank_pattern[name] = r
        alpha_pattern[name] = float(r)

    print(f"Rank pattern: {boosted} layers boosted to {min(base_rank+8, 32)}, "
          f"{len(sorted_layers)-boosted} layers reduced to {max(4, base_rank-4)}")
    return rank_pattern, alpha_pattern
