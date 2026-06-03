import os
import sys
from pathlib import Path

os.environ["HF_HOME"] = "models"

ROOT = Path(__file__).resolve().parent

# 让Python找到src目录
sys.path.insert(
    0,
    str(ROOT / "src")
)

import torch
from PIL import Image
from torchvision import transforms

from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform


# ==================================================
# 数据集路径
# ==================================================

DATA_ROOT = Path(
    r"E:\DataSets\tt100k_2021_paper2"
)

INPUT_DIR = (
    DATA_ROOT /
    "tt100k_60" /
    "images" /
    "train"
)

RAIN_LIST = (
    DATA_ROOT /
    "tt100k_60_weather" /
    "rain.txt"
)

OUTPUT_DIR = (
    DATA_ROOT /
    "tt100k_60_weather" /
    "rain" /
    "images" /
    "train"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# 加载模型（只加载一次）
# ==================================================

print("=" * 60)
print("Loading model...")
print("=" * 60)

model = CycleGAN_Turbo(
    pretrained_name="clear_to_rainy"
)

model.eval()

try:
    model.unet.enable_xformers_memory_efficient_attention()
except:
    print("xformers not found")

# FP16
model.half()

T_val = build_transform(
    "resize_1024x1024"
)

print("Model loaded.")
print()

# ==================================================
# 读取图片列表
# ==================================================

with open(
        RAIN_LIST,
        "r",
        encoding="utf-8") as f:

    image_names = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"Need process: {len(image_names)} images"
)

# ==================================================
# 开始推理
# ==================================================

for idx, stem in enumerate(
        image_names,
        start=1):

    img_path = (
        INPUT_DIR /
        f"{stem}.jpg"
    )

    if not img_path.exists():

        print(
            f"[Skip] {img_path}"
        )

        continue

    try:

        input_image = Image.open(
            img_path
        ).convert("RGB")

        with torch.no_grad():

            input_img = T_val(
                input_image
            )

            x_t = transforms.ToTensor()(
                input_img
            )

            x_t = transforms.Normalize(
                [0.5],
                [0.5]
            )(x_t)

            x_t = (
                x_t
                .unsqueeze(0)
                .cuda()
                .half()
            )

            output = model(x_t)

        output_pil = transforms.ToPILImage()(
            output[0].cpu() * 0.5 + 0.5
        )



        save_path = (
            OUTPUT_DIR /
            img_path.name
        )

        output_pil.save(
            save_path
        )

        print(
            f"[{idx}/{len(image_names)}] "
            f"{img_path.name}"
        )

    except Exception as e:

        print(
            f"[ERROR] {img_path.name}"
        )

        print(e)

print()
print("=" * 60)
print("Finished")
print("=" * 60)