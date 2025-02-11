#!/usr/bin/env python3
import os
import shutil

# 设置源文件夹和目标文件夹的路径
source_folder = "png"  # 替换为你的源文件夹路径
dest_folder = "leadsheet"      # 替换为你的目标文件夹路径

# 如果目标文件夹不存在，则创建
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

ids = []

# 遍历源文件夹下的所有文件
for filename in os.listdir(source_folder):
    # 判断文件名是否以 "-1.png" 结尾
    if filename.endswith("-1.png"):
        id = "-".join(filename.split("-")[:-1])
        if id not in ids:
            ids.append(id)
        # 生成新的文件名，将 "-1.png" 替换为 ".png"
        new_filename = filename[:-len("-1.png")] + ".png"
        src_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, new_filename)
        
        # 拷贝文件到目标文件夹，并重命名
        shutil.copy2(src_path, dest_path)
        print(f"Copied {src_path} to {dest_path}")

print(f"Total {len(ids)} files copied.")