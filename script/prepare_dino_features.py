import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def list_images(image_dir):
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def round_to_patch_multiple(length, patch_size):
    return max(patch_size, (length // patch_size) * patch_size)


def preprocess_image(image, patch_size):
    target_h = round_to_patch_multiple(image.height, patch_size)
    target_w = round_to_patch_multiple(image.width, patch_size)
    transform = transforms.Compose([
        transforms.Resize((target_h, target_w), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image).unsqueeze(0), (image.height, image.width)


def extract_feature_map(model, image_tensor, patch_size):
    with torch.no_grad():
        if hasattr(model, "get_intermediate_layers"):
            try:
                layers = model.get_intermediate_layers(image_tensor, n=1, reshape=True)
                if isinstance(layers, (list, tuple)) and len(layers) > 0:
                    layer = layers[0]
                    if isinstance(layer, tuple):
                        layer = layer[0]
                    if torch.is_tensor(layer) and layer.dim() == 4:
                        return layer.squeeze(0).cpu()
            except TypeError:
                pass

        features = model.forward_features(image_tensor)
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                patch_tokens = features["x_norm_patchtokens"]
            elif "x_prenorm" in features:
                patch_tokens = features["x_prenorm"][:, 1:]
            else:
                raise RuntimeError(
                    "Unsupported DINOv2 feature dict. Expected 'x_norm_patchtokens' or 'x_prenorm'."
                )
        elif torch.is_tensor(features) and features.dim() == 3:
            patch_tokens = features[:, 1:]
        else:
            raise RuntimeError(
                "Unsupported DINOv2 output format. Please adapt extract_feature_map for this model."
            )

        grid_h = image_tensor.shape[-2] // patch_size
        grid_w = image_tensor.shape[-1] // patch_size
        feature_map = patch_tokens.reshape(1, grid_h, grid_w, -1).permute(0, 3, 1, 2).contiguous()
        return feature_map.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description="Prepare offline DINOv2 feature caches for Gaussian Grouping.")
    parser.add_argument("-s", "--source_path", type=str, required=True, help="Scene root path.")
    parser.add_argument("--images", type=str, default="images", help="Image folder under the scene root.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dino_feature",
        help="Output folder under the scene root for cached feature tensors.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="dinov2_vitb14",
        help="torch.hub DINOv2 model name, e.g. dinov2_vits14 / dinov2_vitb14.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for feature extraction.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files.")
    args = parser.parse_args()

    scene_root = Path(args.source_path)
    image_dir = scene_root / args.images
    output_dir = scene_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    try:
        model = torch.hub.load("facebookresearch/dinov2", args.model_name)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load DINOv2 from torch.hub. Ensure internet access or a cached hub checkout is available."
        ) from exc

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    patch_size = int(getattr(model, "patch_size", 14))

    for image_path in tqdm(images, desc="Preparing DINO features"):
        output_path = output_dir / f"{image_path.stem}.pt"
        if output_path.exists() and not args.overwrite:
            continue

        image = Image.open(image_path).convert("RGB")
        image_tensor, image_size = preprocess_image(image, patch_size)
        image_tensor = image_tensor.to(device)
        feature_map = extract_feature_map(model, image_tensor, patch_size)
        torch.save(
            {
                "feature_map": feature_map,
                "image_size": image_size,
                "model_name": args.model_name,
            },
            output_path,
        )


if __name__ == "__main__":
    main()
