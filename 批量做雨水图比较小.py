
import os
import subprocess
from pathlib import Path

# =========================================================
# HuggingFace 镜像（可选）
# =========================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 如果已经缓存成功，可以开启离线
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["DIFFUSERS_OFFLINE"] = "1"

# =========================================================
# 输入输出目录
# =========================================================

INPUT_DIR = Path(
    "/home/jiaoyuqing/datasets/tt100k_2021_paper2/tt100k_60/images/train"
)

OUTPUT_DIR = Path(
    "/home/jiaoyuqing/datasets/tt100k_2021_paper2/tt100k_60_weather/rainytrain"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 支持的图片格式
# =========================================================

IMG_SUFFIX = [".jpg", ".jpeg", ".png", ".bmp"]

# =========================================================
# 收集图片
# =========================================================

img_list = []

for suffix in IMG_SUFFIX:
    img_list.extend(INPUT_DIR.glob(f"*{suffix}"))

img_list = sorted(img_list)

print(f"找到 {len(img_list)} 张图片")

# =========================================================
# 批量生成
# =========================================================

for idx, img_path in enumerate(img_list):

    save_name = img_path.stem + "_rain.png"

    output_path = OUTPUT_DIR / save_name

    # ---------------------------------------------
    # 已存在则跳过
    # ---------------------------------------------

    if output_path.exists():
        print(f"[{idx+1}/{len(img_list)}] 已存在，跳过: {save_name}")
        continue

    print(f"[{idx+1}/{len(img_list)}] 正在处理: {img_path.name}")

    # =====================================================
    # 调用官方推理脚本
    # =====================================================

    cmd = [
        "python",
        "src/inference_unpaired.py",
        "--model_name", "clear_to_rainy",
        "--input_image", str(img_path),
        "--output_dir", str(OUTPUT_DIR),
        "--image_prep", "no_resize"
    ]

    result = subprocess.run(cmd)

    # =====================================================
    # 判断是否成功
    # =====================================================

    if result.returncode == 0:
        print(f"✅ 完成: {img_path.name}")
    else:
        print(f"❌ 失败: {img_path.name}")

print("全部完成")

