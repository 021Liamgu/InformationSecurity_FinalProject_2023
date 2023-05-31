import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


def get_average(root_folder1,root_folder2):
    corpus1 = []
    corpus2 = []
    singlecorpus = []
    hmaveragelist = []
    hmscorelist = []
    GPTaveragelist = []
    GPTscorelist = []


    '''创建字典'''
    for folder_name in os.listdir(root_folder1):
        folder_path = os.path.join(root_folder1, folder_name)

        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            # 在子文件夹中查找data.json文件
            json_file_path = os.path.join(folder_path, "GPTZero.json")
            # 检查data.json文件是否存在
            if os.path.isfile(json_file_path):
                # 打开并读取data.json文件
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        text = item['document']
                        # 分词并去除停用词
                        tokens = text.split()
                        stop_words = set(stopwords.words('english'))

                        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
                        filtered_text = ' '.join(filtered_tokens)

                        corpus1.append(filtered_text)

    vectorizer = TfidfVectorizer()
    # 对文本进行向量化
    hmtfidf_matrix = vectorizer.fit_transform(corpus1)

    for folder_name in os.listdir(root_folder1):
        folder_path = os.path.join(root_folder1, folder_name)

        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            # 在子文件夹中查找data.json文件
            json_file_path = os.path.join(folder_path, "GPTZero.json")
            # 检查data.json文件是否存在
            if os.path.isfile(json_file_path):
                # 打开并读取data.json文件
                '''进行tfidf计算'''
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        singlecorpus.append(item['document'])
                        singletfidf_matrix = vectorizer.transform(singlecorpus)
                        single_average = singletfidf_matrix.data.mean()

                        hmaveragelist.append(single_average)
                        singlecorpus.clear()
                        hmscorelist.append(item['score'])


    for folder_name in os.listdir(root_folder2):
        folder_path = os.path.join(root_folder2, folder_name)

        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            # 在子文件夹中查找data.json文件
            json_file_path = os.path.join(folder_path, "GPTZero.json")
            # 检查data.json文件是否存在
            if os.path.isfile(json_file_path):
                # 打开并读取data.json文件
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        corpus2.append(item['document'])

    vectorizer = TfidfVectorizer()
    # 对文本进行向量化
    GPTtfidf_matrix = vectorizer.fit_transform(corpus2)

    for folder_name in os.listdir(root_folder2):
        folder_path = os.path.join(root_folder2, folder_name)

        # 检查是否为文件夹
        if os.path.isdir(folder_path):
            # 在子文件夹中查找data.json文件
            json_file_path = os.path.join(folder_path, "GPTZero.json")
            # 检查data.json文件是否存在
            if os.path.isfile(json_file_path):
                # 打开并读取data.json文件
                '''进行tfidf计算'''
                with open(json_file_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        singlecorpus.append(item['document'])
                        singletfidf_matrix = vectorizer.transform(singlecorpus)
                        single_average = singletfidf_matrix.data.mean()

                        GPTaveragelist.append(single_average)
                        singlecorpus.clear()
                        GPTscorelist.append(item['score'])
    plt.scatter(hmaveragelist, hmscorelist, label="Human")
    plt.scatter(GPTaveragelist, GPTscorelist, label="GPT")

    # 添加图表标题和轴标签
    plt.title("Scatter Plot")
    plt.xlabel("Average TF-IDF")
    plt.ylabel("Score")

    # 添加图例
    plt.legend()


    plt.show()



hmroot_folder = "./Data_and_Results/Human_Data"
GPTroot_folder = "./Data_and_Results/GPT_Data"

get_average(hmroot_folder,GPTroot_folder)
