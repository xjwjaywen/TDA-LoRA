"""Evaluation metrics."""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from scipy import linalg


class Evaluator:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._load_models()

    def _load_models(self):
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        self.dino = self.dino.to(self.device).eval()
        self.dino_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        import open_clip
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self.clip_model = self.clip_model.to(self.device).eval()
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    @torch.no_grad()
    def _get_dino_features(self, images):
        if isinstance(images[0], (str, Path)):
            images = [Image.open(p).convert("RGB") for p in images]
        tensors = torch.stack([self.dino_transform(img) for img in images]).to(self.device)
        features = []
        for i in range(0, len(tensors), 32):
            features.append(self.dino(tensors[i:i+32]))
        return torch.cat(features, dim=0)

    @torch.no_grad()
    def _get_clip_image_features(self, images):
        if isinstance(images[0], (str, Path)):
            images = [Image.open(p).convert("RGB") for p in images]
        tensors = torch.stack([self.clip_preprocess(img) for img in images]).to(self.device)
        return F.normalize(self.clip_model.encode_image(tensors), dim=-1)

    def compute_dino_similarity(self, gen_images, tgt_images) -> float:
        gen_f = F.normalize(self._get_dino_features(gen_images), dim=-1)
        tgt_f = F.normalize(self._get_dino_features(tgt_images), dim=-1)
        return (gen_f @ tgt_f.mean(0, keepdim=True).T).mean().item()

    def compute_lpips_diversity(self, gen_images) -> float:
        f = F.normalize(self._get_dino_features(gen_images), dim=-1)
        sim = f @ f.T
        mask = ~torch.eye(len(f), dtype=torch.bool, device=self.device)
        return (1 - sim[mask]).mean().item()

    def compute_clip_score(self, gen_images, prompt: str) -> float:
        img_f = self._get_clip_image_features(gen_images)
        txt_f = F.normalize(self.clip_model.encode_text(
            self.clip_tokenizer([prompt]).to(self.device)), dim=-1)
        return (img_f @ txt_f.T).mean().item()

    def compute_fid(self, gen_images, ref_images) -> float:
        g = self._get_dino_features(gen_images).cpu().numpy()
        r = self._get_dino_features(ref_images).cpu().numpy()
        mu_g, sig_g = g.mean(0), np.cov(g, rowvar=False)
        mu_r, sig_r = r.mean(0), np.cov(r, rowvar=False)
        diff = mu_g - mu_r
        covmean, _ = linalg.sqrtm(sig_g @ sig_r, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sig_g + sig_r - 2 * covmean))

    def evaluate_all(self, gen_images, tgt_images, prompt: str) -> dict:
        r = {
            "dino_similarity": self.compute_dino_similarity(gen_images, tgt_images),
            "lpips_diversity": self.compute_lpips_diversity(gen_images),
            "clip_score": self.compute_clip_score(gen_images, prompt),
            "fid": self.compute_fid(gen_images, tgt_images),
        }
        print("\n=== Evaluation Results ===")
        for k, v in r.items():
            print(f"  {k}: {v:.4f}")
        return r
