import os
import subprocess
# os.environ["HF_HOME"] = "models"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["DIFFUSERS_OFFLINE"] = "1"

subprocess.run([
    "python",
    "src/inference_unpaired.py",
    "--model_name", "clear_to_rainy",
    # "--model_name", "clear_to_foggy",
    "--input_image",
    "/home/jiaoyuqing/tt100k_2021/yolojack/images/train/165.jpg",
    "--output_dir",
    "outputs_test"
])