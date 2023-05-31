import os
import json
import matplotlib.pyplot as plt


if os.path.isdir(folder_path1):
    # 在子文件夹中查找data.json文件
    json_file_path = os.path.join(folder_path1, "GPTZero.json")
    # 检查data.json文件是否存在
    if os.path.isfile(json_file_path):
        # 打开并读取data.json文件
        with open(json_file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                purewholecorpus.append(item['document'])