
import os

os.environ["HF_HOME"] = r"models"

import subprocess

subprocess.run([
    "python",
    "src/inference_unpaired.py",
    "--model_name", "clear_to_rainy",
    "--input_image",
    r"E:\DataSets\tt100k_2021_paper2\tt100k_60\images\train\35.jpg",
    "--output_dir",
    "outputs_test"
])