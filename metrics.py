from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import traceback

loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []

    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for fname in sorted(os.listdir(renders_dir)):
        if Path(fname).suffix.lower() not in valid_ext:
            continue

        render = Image.open(renders_dir / fname).convert("RGB")
        gt = Image.open(gt_dir / fname).convert("RGB")

        render_tensor = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()

        renders.append(render_tensor)
        gts.append(gt_tensor)
        image_names.append(fname)

    return renders, gts, image_names


def evaluate(model_paths):
    full_dict = {}
    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"
            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {test_dir}")

            for method in sorted(os.listdir(test_dir)):
                method_dir = test_dir / method
                if not method_dir.is_dir():
                    continue

                print("Method:", method)

                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"

                if not gt_dir.exists():
                    raise FileNotFoundError(f"GT directory not found: {gt_dir}")
                if not renders_dir.exists():
                    raise FileNotFoundError(f"Render directory not found: {renders_dir}")

                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssim_val = ssim(renders[idx], gts[idx]).item()
                    psnr_val = psnr(renders[idx], gts[idx]).item()

                    # LPIPS 官方更推荐输入 [-1, 1]
                    render_lpips = renders[idx] * 2 - 1
                    gt_lpips = gts[idx] * 2 - 1
                    lpips_val = loss_fn_vgg(render_lpips, gt_lpips).mean().item()

                    ssims.append(ssim_val)
                    psnrs.append(psnr_val)
                    lpipss.append(lpips_val)

                ssim_mean = sum(ssims) / len(ssims)
                psnr_mean = sum(psnrs) / len(psnrs)
                lpips_mean = sum(lpipss) / len(lpipss)

                print(f"  SSIM : {ssim_mean:>12.7f}")
                print(f"  PSNR : {psnr_mean:>12.7f}")
                print(f"  LPIPS: {lpips_mean:>12.7f}")
                print("")

                full_dict[scene_dir][method] = {
                    "SSIM": ssim_mean,
                    "PSNR": psnr_mean,
                    "LPIPS": lpips_mean,
                }

                per_view_dict[scene_dir][method] = {
                    "SSIM": {name: val for name, val in zip(image_names, ssims)},
                    "PSNR": {name: val for name, val in zip(image_names, psnrs)},
                    "LPIPS": {name: val for name, val in zip(image_names, lpipss)},
                }

            with open(scene_dir + "/results.json", "w") as fp:
                json.dump(full_dict[scene_dir], fp, indent=2)
            with open(scene_dir + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=2)

        except Exception:
            traceback.print_exc()
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    args = parser.parse_args()
    evaluate(args.model_paths)