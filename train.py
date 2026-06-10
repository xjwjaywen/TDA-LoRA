"""Main training script for TDA-LoRA."""
import argparse
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from src.tda_lora import (
    create_tda_lora, TimestepSampler, compute_domain_gap,
    compute_layer_importance, build_rank_pattern,
)
from src.dataset import FewShotDataset, DATASET_LOADERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--class_name", type=str, default=None)
    parser.add_argument("--num_shots", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--method", type=str, default="tda_lora",
                        choices=["tda_lora", "lora_only", "timestep_only",
                                 "domain_only", "layer_only", "td_no_layer",
                                 "tl_no_domain", "dl_no_timestep"])
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.dataset:
        config["dataset"]["name"] = args.dataset
    if args.class_name:
        config["dataset"]["class_name"] = args.class_name
    if args.num_shots:
        config["dataset"]["num_shots"] = args.num_shots
    if args.model_path:
        config["model"]["pretrained_model"] = args.model_path

    dataset_name = config["dataset"]["name"]
    class_name = config["dataset"]["class_name"]
    num_shots = config["dataset"]["num_shots"]
    method = args.method
    model_id = config["model"]["pretrained_model"]
    lora_cfg = config["lora"]
    tda_cfg = config["tda"]
    train_cfg = config["training"]

    config["output"]["output_dir"] = f"./outputs/{dataset_name}/{class_name}/{method}_shot{num_shots}"
    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TDA-LoRA | method={method} | {dataset_name}/{class_name} | {num_shots}-shot")
    print(f"{'='*60}")

    # --- Load data ---
    loader_fn = DATASET_LOADERS[dataset_name]
    real_paths, concept = loader_fn(config["dataset"]["data_dir"], class_name, num_shots)
    if not real_paths:
        return

    prompt = f"a photo of a {concept}"
    dataset = FewShotDataset(real_paths, prompt, config["model"]["resolution"])
    dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"],
                            shuffle=True, num_workers=2, pin_memory=True)

    # --- Load base model ---
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(args.device)
    vae = AutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float16
    ).to(args.device)
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to(args.device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # --- Determine components based on method ---
    use_timestep = method in ("tda_lora", "timestep_only", "td_no_layer", "tl_no_domain")
    use_domain = method in ("tda_lora", "domain_only", "td_no_layer", "dl_no_timestep")
    use_layer = method in ("tda_lora", "layer_only", "tl_no_domain", "dl_no_timestep")

    # --- Domain gap ---
    domain_gap = compute_domain_gap(real_paths, model_id, args.device) if use_domain else 0.0

    # --- Layer importance (requires initial LoRA for warmup) ---
    rank_pattern = {}
    alpha_pattern = {}
    if use_layer:
        # Create temporary LoRA for warmup
        temp_unet = create_tda_lora(unet, lora_cfg["target_modules"],
                                     base_rank=lora_cfg["base_rank"], alpha=lora_cfg["alpha"])
        importance = compute_layer_importance(
            temp_unet, dataloader, noise_scheduler, text_encoder, vae, tokenizer,
            warmup_steps=tda_cfg["warmup_steps"], device=args.device,
        )
        # Remove temp LoRA
        temp_unet = temp_unet.unload()

        rank_pattern, alpha_pattern = build_rank_pattern(
            importance, lora_cfg["base_rank"], domain_gap,
            tda_cfg["domain_scale_factor"], tda_cfg["layer_importance_top_k"],
        )
    elif use_domain:
        # Domain-only: uniform rank adjusted by domain gap
        effective_rank = max(2, int(lora_cfg["base_rank"] * (1 - tda_cfg["domain_scale_factor"] * domain_gap)))
        print(f"Domain-only: all layers rank={effective_rank}")
        # rank_pattern stays empty, just adjust base_rank
        lora_cfg["base_rank"] = effective_rank

    # --- Create final LoRA ---
    unet = create_tda_lora(unet, lora_cfg["target_modules"],
                            base_rank=lora_cfg["base_rank"], alpha=lora_cfg["alpha"],
                            rank_pattern=rank_pattern, alpha_pattern=alpha_pattern)

    # --- Timestep sampler ---
    ts_sampler = TimestepSampler(
        num_timesteps=noise_scheduler.config.num_train_timesteps,
        strategy="tda" if use_timestep else "uniform",
        domain_gap=domain_gap,
    )
    print(f"Timestep strategy: {'tda' if use_timestep else 'uniform'}")

    # --- Training ---
    optimizer = torch.optim.AdamW(
        [p for p in unet.parameters() if p.requires_grad],
        lr=train_cfg["learning_rate"],
    )

    unet.train()
    data_iter = iter(dataloader)
    progress = tqdm(range(train_cfg["num_steps"]), desc=f"Training {method}")
    running_loss = 0.0

    for step in progress:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        pixel_values = batch["pixel_values"].to(args.device, dtype=torch.float16)
        batch_prompt = batch["prompt"][0]

        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
            tokens = tokenizer(
                [batch_prompt] * latents.shape[0], padding="max_length",
                max_length=tokenizer.model_max_length, truncation=True,
                return_tensors="pt",
            ).input_ids.to(args.device)
            encoder_hidden_states = text_encoder(tokens)[0]

        noise = torch.randn_like(latents)
        timesteps = ts_sampler.sample(latents.shape[0], args.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
        loss = F.mse_loss(noise_pred.float(), noise.float())
        loss = loss / train_cfg["gradient_accumulation"]
        loss.backward()

        if (step + 1) % train_cfg["gradient_accumulation"] == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in unet.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * train_cfg["gradient_accumulation"]

        if (step + 1) % config["output"]["log_every"] == 0:
            avg_loss = running_loss / config["output"]["log_every"]
            progress.set_postfix(loss=f"{avg_loss:.4f}")
            running_loss = 0.0

        if (step + 1) % config["output"]["save_every"] == 0:
            save_path = output_dir / f"checkpoint-{step+1}"
            unet.save_pretrained(str(save_path))

    final_path = output_dir / "final"
    unet.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")

    # --- Generate & Evaluate ---
    print("\nGenerating evaluation images...")
    unet.eval()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, unet=unet, torch_dtype=torch.float16
    ).to(args.device)

    eval_cfg = config["evaluation"]
    generator = torch.Generator(device=args.device)
    images = []
    for i in range(eval_cfg["num_eval_images"]):
        generator.manual_seed(train_cfg["seed"] + i)
        img = pipe(prompt, num_inference_steps=30, guidance_scale=7.5,
                   generator=generator).images[0]
        images.append(img)

    gen_dir = output_dir / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        img.save(gen_dir / f"gen_{i:04d}.png")

    del pipe
    torch.cuda.empty_cache()

    from src.evaluate import Evaluator
    evaluator = Evaluator(device=args.device)
    results = evaluator.evaluate_all(images, real_paths, prompt)

    with open(output_dir / "metrics.txt", "w") as f:
        f.write(f"method: {method}\ndataset: {dataset_name}/{class_name}\nnum_shots: {num_shots}\n")
        f.write(f"domain_gap: {domain_gap:.4f}\n")
        f.write(f"use_timestep: {use_timestep}\nuse_domain: {use_domain}\nuse_layer: {use_layer}\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.6f}\n")

    print(f"\nResults saved to {output_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
