import glob
import shutil
import os

"""
img path 변경 & 이름 수정
img_path, dest_dir 개인에 맞게 수정 필요
"""

img_path = "../../project1/data/train/images/"

img_files = glob.glob(img_path + "/**/*.jpeg", recursive=True) + glob.glob(
    img_path + "/**/*.jpg", recursive=True
)

dest_dir = "../../images"

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

for img_file in img_files:
    filename = img_file.split("\\")[-2].split("_")[0] + "_" + img_file.split("\\")[-1]
    shutil.copy2(img_file, os.path.join(dest_dir, filename))
