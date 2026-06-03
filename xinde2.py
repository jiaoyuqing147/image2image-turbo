
import os
import subprocess

# os.environ["HF_HOME"] = "models"

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

subprocess.run([
    "python",
    "src/inference_unpaired.py",

    "--model_name", "clear_to_rainy",

    "--input_image",
    "/home/jiaoyuqing/tt100k_2021/yolojack/images/train/165.jpg",

    "--output_dir",
    "outputs_test",

    # 关键：不resize
    "--image_prep", "no_resize"
])

