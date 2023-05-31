import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


root_folder1 = ''
corpus1 = []
corpus2 = []
# 遍历根文件夹下的所有子文件夹
for folder_name in os.listdir(root_folder1):
    folder_path = os.path.join(root_folder1, folder_name)

    # 检查是否为文件夹
    if os.path.isdir(folder_path):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path, "data.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    corpus1.append(item['document'])
vectorizer = TfidfVectorizer()
# 对文本进行向量化
hmtfidf_matrix = vectorizer.fit_transform(corpus1)
print(hmtfidf_matrix.data)
hmaverage_tfidf = hmtfidf_matrix.data.mean()
print("Average TF-IDF score:", hmaverage_tfidf)