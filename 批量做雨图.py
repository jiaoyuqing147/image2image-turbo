
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(ROOT / "src")
)

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform



# =========================================================
# HuggingFace（可选）
# =========================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 已缓存模型后推荐开启
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["DIFFUSERS_OFFLINE"] = "1"

# =========================================================
# 输入输出路径
# =========================================================

INPUT_DIR = Path(
    "/home/jiaoyuqing/tt100k_2021/yolojack/images/train"
)

OUTPUT_DIR = Path(
    "/home/jiaoyuqing/tt100k_2021_weather/rainy/images/train"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 模型
# =========================================================

DEVICE = "cuda"

print("Loading model...")

model = CycleGAN_Turbo(
    pretrained_name="clear_to_rainy"
).to(DEVICE)

model.eval()

print("Model loaded.")

# =========================================================
# transform
# =========================================================

transform = build_transform("no_resize")

# =========================================================
# 图片列表
# =========================================================

IMG_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp"]

img_list = []

for suffix in IMG_SUFFIX:
    img_list.extend(INPUT_DIR.glob(f"*{suffix}"))

img_list = sorted(img_list)

print(f"Found {len(img_list)} images")

# =========================================================
# 批量处理
# =========================================================

for img_path in tqdm(img_list):

    save_name = img_path.stem + "_rain.png"

    save_path = OUTPUT_DIR / save_name

    # -----------------------------------------------------
    # 已存在则跳过
    # -----------------------------------------------------

    if save_path.exists():
        continue

    try:

        # =================================================
        # 读取图片
        # =================================================

        img = Image.open(img_path).convert("RGB")

        # =================================================
        # 对齐到8（Stable Diffusion要求）
        # =================================================

        w, h = img.size

        new_w = (w // 8) * 8
        new_h = (h // 8) * 8

        if new_w != w or new_h != h:
            img = img.resize((new_w, new_h))

        # =================================================
        # transform
        # =================================================

        x = transform(img)

        x = transforms.ToTensor()(x)
        x = x.unsqueeze(0).to(DEVICE)

        x = x * 2 - 1

        # =================================================
        # 推理
        # =================================================

        with torch.no_grad():

            output = model(x)

        # =================================================
        # 保存
        # =================================================

        output = output[0].cpu().clamp(-1, 1)

        output = (output + 1) / 2

        out_pil = transforms.ToPILImage()(output)

        out_pil.save(save_path)

    except Exception as e:

        print(f"\nERROR: {img_path.name}")
        print(e)

print("All done.")

