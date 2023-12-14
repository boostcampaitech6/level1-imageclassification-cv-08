import glob
import shutil
import os

"""
train 폴더는 해당 레포랑 같은 레벨에 있어야 함
"""

img_path = "../train/images"

img_files = glob.glob(img_path + "/**/*.jpeg", recursive=True) + glob.glob(
    img_path + "/**/*.jpg", recursive=True
)

dest_dir = "../images"

if not os.path.exists(dest_dir):
    os.mkdir(dest_dir)

for img_file in img_files:
    filename_list = os.path.split(img_file)
    filename = filename_list[0].split(os.sep)[-1].split("_")[0] + "_" + filename_list[1]
    shutil.copy2(img_file, os.path.join(dest_dir, filename))
