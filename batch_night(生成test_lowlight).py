
import os
import sys
from pathlib import Path

# ==================================================
# ROOT
# ==================================================

ROOT = Path(__file__).resolve().parent

# ==================================================
# HuggingFace 本地缓存 + 离线模式
# ==================================================

os.environ["HF_HOME"] = str(ROOT / "models")

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["DIFFUSERS_OFFLINE"] = "1"

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
# 数据集路径（Linux服务器）
# ==================================================

DATA_ROOT = Path(
    "/home/jiaoyuqing/datasets/tt100k_2021_paper2"
)

INPUT_DIR = (
    DATA_ROOT /
    "tt100k_60" /
    "images" /
    "test"
)

OUTPUT_DIR = (
    DATA_ROOT /
    "tt100k_60_weather" /
    "test_lowlight"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 60)
print("Dataset Paths")
print("=" * 60)
print(f"INPUT_DIR  : {INPUT_DIR}")
print(f"OUTPUT_DIR : {OUTPUT_DIR}")
print()

# ==================================================
# CUDA信息
# ==================================================

print("=" * 60)
print("CUDA Info")
print("=" * 60)

print(
    "CUDA Available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

print()

# ==================================================
# 加载模型（只加载一次）
# ==================================================

print("=" * 60)
print("Loading day_to_night model...")
print("=" * 60)

model = CycleGAN_Turbo(
    pretrained_name="day_to_night"
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
# 读取 test 文件夹全部图像
# ==================================================

image_paths = sorted(
    list(INPUT_DIR.glob("*.jpg")) +
    list(INPUT_DIR.glob("*.png")) +
    list(INPUT_DIR.glob("*.jpeg"))
)

print(
    f"Need process: {len(image_paths)} images"
)

# 测试时打开
# image_paths = image_paths[:10]

# ==================================================
# 开始推理
# ==================================================

for idx, img_path in enumerate(
        image_paths,
        start=1):

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

            output = model(
                x_t,
                caption="driving at night"
            )

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
            f"[{idx}/{len(image_paths)}] "
            f"{img_path.name}"
        )

    except Exception as e:

        print(
            f"[ERROR] {img_path.name}"
        )

        print(e)

print()
print("=" * 60)
print("Test Lowlight Generation Finished")
print("=" * 60)

