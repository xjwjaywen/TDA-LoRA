"""
TDA-LoRA: Timestep-Domain Adaptive Low-Rank Adaptation.

Core idea: different timesteps and domains need different adaptation capacities.
Instead of fixed rank across all layers, we assign ranks based on:
  1. Timestep bin (high/mid/low noise)
  2. Domain gap (how far target domain is from pretrained distribution)
  3. Layer importance (gradient-guided)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple


class TDALoRALinear(nn.Module):
    """A LoRA layer with timestep-adaptive rank scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int = 8,
        alpha: float = 16.0,
        num_timestep_bins: int = 3,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.alpha = alpha
        self.scaling = alpha / max_rank
        self.num_timestep_bins = num_timestep_bins

        self.lora_A = nn.Parameter(torch.zeros(max_rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, max_rank))

        # Per-timestep-bin gate: learns how much of each rank dimension to use
        self.timestep_gates = nn.Parameter(torch.ones(num_timestep_bins, max_rank))

        # Layer importance score (set externally after warmup)
        self.importance = 1.0

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor, timestep_bin: int = 1) -> torch.Tensor:
        gate = torch.sigmoid(self.timestep_gates[timestep_bin])
        # Scale gate by layer importance
        gate = gate * self.importance

        # Gated low-rank update: B @ diag(gate) @ A
        adapted_A = self.lora_A * gate.unsqueeze(1)  # (rank, in) * (rank, 1)
        delta = F.linear(x, adapted_A)
        delta = F.linear(delta, self.lora_B)

        return delta * self.scaling


class TDALoRAUNet(nn.Module):
    """Wraps a UNet with TDA-LoRA layers."""

    def __init__(
        self,
        unet: nn.Module,
        target_modules: List[str],
        base_rank: int = 8,
        alpha: float = 16.0,
        num_timestep_bins: int = 3,
        domain_gap: float = 0.0,
        domain_scale_factor: float = 0.3,
    ):
        super().__init__()
        self.unet = unet
        self.num_timestep_bins = num_timestep_bins
        self.current_timestep_bin = 1
        self.tda_layers: Dict[str, TDALoRALinear] = {}

        # Freeze base model
        for param in unet.parameters():
            param.requires_grad = False

        # Domain-adjusted rank
        effective_rank = max(2, int(base_rank * (1 - domain_scale_factor * domain_gap)))
        print(f"Domain gap: {domain_gap:.3f} -> effective rank: {effective_rank} (base: {base_rank})")

        # Inject TDA-LoRA layers
        self._inject_lora(target_modules, effective_rank, alpha, num_timestep_bins)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in self.parameters())
        print(f"TDA-LoRA: trainable={total_params:,} / total={total_all:,} ({100*total_params/total_all:.2f}%)")

    def _inject_lora(self, target_modules, rank, alpha, num_bins):
        for name, module in self.unet.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(t in name for t in target_modules):
                continue

            tda_layer = TDALoRALinear(
                module.in_features, module.out_features,
                max_rank=rank, alpha=alpha, num_timestep_bins=num_bins,
            ).to(module.weight.device, dtype=module.weight.dtype)

            self.tda_layers[name] = tda_layer
            # Register as submodule for optimizer
            safe_name = name.replace(".", "_")
            self.register_module(f"tda_{safe_name}", tda_layer)

            # Hook to add LoRA output
            self._register_hook(module, tda_layer)

        print(f"Injected TDA-LoRA into {len(self.tda_layers)} layers")

    def _register_hook(self, original_module, tda_layer):
        def hook(module, input, output):
            x = input[0]
            lora_out = tda_layer(x.to(tda_layer.lora_A.dtype), self.current_timestep_bin)
            return output + lora_out.to(output.dtype)
        original_module.register_forward_hook(hook)

    def set_timestep_bin(self, timestep: int, total_timesteps: int = 1000):
        """Map a diffusion timestep to a bin index."""
        ratio = timestep / total_timesteps
        if ratio > 0.66:
            self.current_timestep_bin = 0  # high noise
        elif ratio > 0.33:
            self.current_timestep_bin = 1  # mid noise
        else:
            self.current_timestep_bin = 2  # low noise

    def forward(self, noisy_latents, timesteps, encoder_hidden_states):
        # Set bin based on mean timestep in batch
        mean_t = timesteps.float().mean().item()
        self.set_timestep_bin(int(mean_t), self.unet.config.get("num_train_timesteps", 1000))
        return self.unet(noisy_latents, timesteps, encoder_hidden_states)

    def save_tda_lora(self, path):
        """Save only TDA-LoRA parameters."""
        state = {}
        for name, layer in self.tda_layers.items():
            state[f"{name}.lora_A"] = layer.lora_A.data.cpu()
            state[f"{name}.lora_B"] = layer.lora_B.data.cpu()
            state[f"{name}.timestep_gates"] = layer.timestep_gates.data.cpu()
            state[f"{name}.importance"] = torch.tensor(layer.importance)
        torch.save(state, path)
        print(f"TDA-LoRA saved to {path} ({len(state)} tensors)")


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

    # Target image features
    images = [Image.open(p).convert("RGB") for p in target_image_paths]
    img_tensors = torch.stack([preprocess(img) for img in images]).to(device)
    with torch.no_grad():
        img_features = model.encode_image(img_tensors)
        img_features = F.normalize(img_features, dim=-1)
        target_centroid = img_features.mean(dim=0)

    # Reference: "a photo" feature as proxy for pretrained distribution center
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

    # Normalize to [0, 1] range (typical gaps are 0.2-0.8)
    gap = max(0.0, min(1.0, gap / 0.8))
    print(f"Domain gap score: {gap:.3f}")
    return gap


def compute_layer_importance(
    tda_unet: TDALoRAUNet,
    dataloader,
    noise_scheduler,
    text_encoder,
    vae,
    tokenizer,
    warmup_steps: int = 50,
    top_k_ratio: float = 0.5,
    device: str = "cuda",
):
    """Gradient-guided layer importance estimation."""
    print(f"Computing layer importance ({warmup_steps} warmup steps)...")
    grad_norms = {name: 0.0 for name in tda_unet.tda_layers}

    tda_unet.train()
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

        noise_pred = tda_unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss.backward()

        for name, layer in tda_unet.tda_layers.items():
            if layer.lora_A.grad is not None:
                grad_norms[name] += layer.lora_A.grad.norm().item()
                grad_norms[name] += layer.lora_B.grad.norm().item()

        tda_unet.zero_grad()

    # Normalize and assign importance
    max_norm = max(grad_norms.values()) if grad_norms else 1.0
    sorted_layers = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)
    top_k = int(len(sorted_layers) * top_k_ratio)

    for i, (name, norm) in enumerate(sorted_layers):
        if i < top_k:
            tda_unet.tda_layers[name].importance = 1.0
        else:
            tda_unet.tda_layers[name].importance = 0.3  # reduced but not zero
        normalized = norm / max_norm if max_norm > 0 else 0
        print(f"  {name}: grad_norm={normalized:.3f} importance={tda_unet.tda_layers[name].importance}")

    print(f"Top-{top_k} layers get full importance, rest reduced to 0.3")
