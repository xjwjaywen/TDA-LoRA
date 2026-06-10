"""Few-shot dataset loading."""
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FewShotDataset(Dataset):
    def __init__(self, image_paths: list, prompt: str, resolution: int = 512):
        self.paths = image_paths
        self.prompt = prompt
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return max(len(self.paths) * 100, 1000)

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]
        image = Image.open(path).convert("RGB")
        return {"pixel_values": self.transform(image), "prompt": self.prompt}


def prepare_cub200(data_dir: str, class_name: str, num_shots: int, seed: int = 42):
    data_path = Path(data_dir) / "CUB_200_2011" / "images"
    if not data_path.exists():
        print(f"CUB-200 not found at {data_path}")
        return [], class_name

    for d in sorted(data_path.iterdir()):
        if class_name.lower() in d.name.lower():
            all_images = sorted(d.glob("*.jpg"))
            selected = random.Random(seed).sample(all_images, min(num_shots, len(all_images)))
            concept = class_name.replace("_", " ").split(".")[-1].strip()
            print(f"CUB-200: loaded {len(selected)} images for '{concept}'")
            return [str(p) for p in selected], concept

    print(f"Class '{class_name}' not found")
    return [], class_name


def prepare_mvtec(data_dir: str, class_name: str, num_shots: int, seed: int = 42):
    data_path = Path(data_dir) / "mvtec_anomaly_detection" / class_name / "train" / "good"
    if not data_path.exists():
        print(f"MVTec AD not found at {data_path}")
        return [], class_name

    all_images = sorted(data_path.glob("*.png"))
    selected = random.Random(seed).sample(all_images, min(num_shots, len(all_images)))
    print(f"MVTec AD: loaded {len(selected)} images for '{class_name}'")
    return [str(p) for p in selected], class_name


DATASET_LOADERS = {"cub200": prepare_cub200, "mvtec": prepare_mvtec}
