
import os
import subprocess
from pathlib import Path

# =========================================================
# HuggingFace 镜像
# =========================================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 不要开启离线
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["DIFFUSERS_OFFLINE"] = "1"

# =========================================================
# 路径
# =========================================================

INPUT_DIR = Path(
    "/home/jiaoyuqing/datasets/tt100k_2021_paper2/tt100k_60/images/train"
)

RAIN_TXT = Path(
    "/home/jiaoyuqing/datasets/tt100k_2021_paper2/tt100k_60_weather/rain.txt"
)

OUTPUT_DIR = Path(
    "/home/jiaoyuqing/datasets/tt100k_2021_paper2/tt100k_60_weather/rainytrain"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 读取 rain.txt
# =========================================================

with open(RAIN_TXT, "r", encoding="utf-8") as f:

    img_names = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

print(f"需要处理 {len(img_names)} 张图片")

# =========================================================
# 批量处理
# =========================================================

for idx, img_name in enumerate(img_names):

    # =====================================================
    # 输入图片路径
    # =====================================================

    img_path = INPUT_DIR / f"{img_name}.jpg"

    # =====================================================
    # 检查输入图片是否存在
    # =====================================================

    if not img_path.exists():

        print(f"❌ 原图不存在: {img_name}")

        continue

    # =====================================================
    # 输出文件
    # =====================================================

    save_name = f"{img_name}_rain.png"

    output_path = OUTPUT_DIR / save_name

    # =====================================================
    # 已生成则跳过
    # =====================================================

    if output_path.exists():

        print(
            f"[{idx+1}/{len(img_names)}] "
            f"已生成，跳过: {save_name}"
        )

        continue

    print(
        f"[{idx+1}/{len(img_names)}] "
        f"正在处理: {img_name}"
    )

    # =====================================================
    # 调用官方推理脚本
    # =====================================================

    cmd = [
        "python",
        "src/inference_unpaired.py",

        "--model_name", "clear_to_rainy",

        "--input_image",
        str(img_path),

        "--output_dir",
        str(OUTPUT_DIR),

        # =================================================
        # 1024增强（推荐）
        # =================================================

        "--image_prep", "resize_1024x1024"

        # 如果想原图：
        # "--image_prep", "no_resize"
    ]

    result = subprocess.run(cmd)

    # =====================================================
    # 判断是否成功
    # =====================================================

    if result.returncode == 0:

        print(f"✅ 完成: {img_name}")

    else:

        print(f"❌ 失败: {img_name}")

print("全部完成")

