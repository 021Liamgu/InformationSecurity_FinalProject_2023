import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


def get_average(root_folder1,root_folder2):
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


    for folder_name in os.listdir(root_folder2):
        folder_path = os.path.join(root_folder2, folder_name)

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
                        corpus2.append(item['document'])

    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer()
    # 对文本进行向量化
    GPTtfidf_matrix = vectorizer.fit_transform(corpus2)
    print(GPTtfidf_matrix.data)

    GPTaverage_tfidf = GPTtfidf_matrix.data.mean()
    print("Average TF-IDF score:", GPTaverage_tfidf)

    plt.boxplot(hmaverage_tfidf)
    plt.show()
    # plt.hist(GPTtfidf_matrix.data, bins=100, alpha=0.5, label='GPT')
    # plt.xlabel("Vocabulary TF-IDF")
    # plt.ylabel("Frequency")
    # plt.title("Histogram of TF-IDF")
    # plt.legend()
    # plt.show()



hmroot_folder = "./Data_and_Results/Human_Data"
GPTroot_folder = "./Data_and_Results/GPT_Data"

get_average(hmroot_folder,GPTroot_folder)
