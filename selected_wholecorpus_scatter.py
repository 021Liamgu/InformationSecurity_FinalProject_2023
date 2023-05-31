import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords



def get_average(folder_path1,folder_path2,folder_path3,folder_path4):
    corpus1 = []
    corpus2 = []
    corpus3 = []
    corpus4 = []
    purewholecorpus = []
    processedcorpus = []
    corpus5 = []
    singlecorpus = []
    TOEFLaveragelist = []
    TOEFLscorelist = []
    GRADE8averagelist = []
    GRADE8scorelist = []
    COLLEGEaveragelist = []
    COLLEGEscorelist = []
    COLLEGEAIaveragelist = []
    COLLEGEAIscorelist = []

    '''build up wholecorpus'''
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
    if os.path.isdir(folder_path2):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path2, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    purewholecorpus.append(item['document'])
    if os.path.isdir(folder_path3):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path3, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    purewholecorpus.append(item['document'])
    if os.path.isdir(folder_path4):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path4, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    purewholecorpus.append(item['document'])

    # 分词并去除停用词
    for text in purewholecorpus:

        tokens = text.split()
        stop_words = set(stopwords.words('english'))

        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        filtered_text = ' '.join(filtered_tokens)

        processedcorpus.append(filtered_text)

    vectorizer = TfidfVectorizer()
    # 对文本进行向量化
    hmtfidf_matrix = vectorizer.fit_transform(processedcorpus)






    '''TOEFL'''


    # 检查是否为文件夹
    # if os.path.isdir(folder_path1):
    #     # 在子文件夹中查找data.json文件
    #     json_file_path = os.path.join(folder_path1, "GPTZero.json")
    #     # 检查data.json文件是否存在
    #     if os.path.isfile(json_file_path):
    #         # 打开并读取data.json文件
    #         with open(json_file_path, 'r') as f:
    #             data = json.load(f)
    #             for item in data:
    #                 text = item['document']
    #                 # 分词并去除停用词
    #                 tokens = text.split()
    #                 stop_words = set(stopwords.words('english'))
    #
    #                 filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #                 filtered_text = ' '.join(filtered_tokens)
    #
    #                 corpus1.append(filtered_text)

    # 检查是否为文件夹
    if os.path.isdir(folder_path1):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path1, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            '''进行tfidf计算'''
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    text = item['document']
                    # 分词并去除停用词
                    tokens = text.split()
                    stop_words = set(stopwords.words('english'))

                    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
                    filtered_text = ' '.join(filtered_tokens)

                    singlecorpus.append(filtered_text)

                    singletfidf_matrix = vectorizer.transform(singlecorpus)
                    single_average = singletfidf_matrix.data.mean()

                    TOEFLaveragelist.append(single_average)
                    singlecorpus.clear()
                    TOEFLscorelist.append(item['score'])

    '''GRADE8'''
    # if os.path.isdir(folder_path2):
    #     # 在子文件夹中查找data.json文件
    #     json_file_path = os.path.join(folder_path2, "GPTZero.json")
    #     # 检查data.json文件是否存在
    #     if os.path.isfile(json_file_path):
    #         # 打开并读取data.json文件
    #         with open(json_file_path, 'r') as f:
    #             data = json.load(f)
    #             for item in data:
    #                 text = item['document']
    #                 # 分词并去除停用词
    #                 tokens = text.split()
    #                 stop_words = set(stopwords.words('english'))
    #
    #                 filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #                 filtered_text = ' '.join(filtered_tokens)
    #
    #                 corpus2.append(filtered_text)
    #
    # vectorizer = TfidfVectorizer()
    # # 对文本进行向量化
    # hmtfidf_matrix = vectorizer.fit_transform(corpus2)

    # 检查是否为文件夹
    if os.path.isdir(folder_path2):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path2, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            '''进行tfidf计算'''
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    text = item['document']
                    # 分词并去除停用词
                    tokens = text.split()
                    stop_words = set(stopwords.words('english'))

                    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
                    filtered_text = ' '.join(filtered_tokens)

                    singlecorpus.append(filtered_text)
                    singletfidf_matrix = vectorizer.transform(singlecorpus)
                    single_average = singletfidf_matrix.data.mean()

                    GRADE8averagelist.append(single_average)
                    singlecorpus.clear()
                    GRADE8scorelist.append(item['score'])



    '''College'''
    # if os.path.isdir(folder_path3):
    #     # 在子文件夹中查找data.json文件
    #     json_file_path = os.path.join(folder_path3, "GPTZero.json")
    #     # 检查data.json文件是否存在
    #     if os.path.isfile(json_file_path):
    #         # 打开并读取data.json文件
    #         with open(json_file_path, 'r') as f:
    #             data = json.load(f)
    #             for item in data:
    #                 text = item['document']
    #                 # 分词并去除停用词
    #                 tokens = text.split()
    #                 stop_words = set(stopwords.words('english'))
    #
    #                 filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #                 filtered_text = ' '.join(filtered_tokens)
    #
    #                 corpus3.append(filtered_text)
    #
    # vectorizer = TfidfVectorizer()
    # # 对文本进行向量化
    # hmtfidf_matrix = vectorizer.fit_transform(corpus3)

    # 检查是否为文件夹
    if os.path.isdir(folder_path3):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path3, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            '''进行tfidf计算'''
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    text = item['document']
                    # 分词并去除停用词
                    tokens = text.split()
                    stop_words = set(stopwords.words('english'))

                    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
                    filtered_text = ' '.join(filtered_tokens)

                    singlecorpus.append(item['document'])
                    singletfidf_matrix = vectorizer.transform(singlecorpus)
                    single_average = singletfidf_matrix.data.mean()

                    COLLEGEaveragelist.append(single_average)
                    singlecorpus.clear()
                    COLLEGEscorelist.append(item['score'])

    '''COLLEGEAI'''

    # if os.path.isdir(folder_path4):
    #     # 在子文件夹中查找data.json文件
    #     json_file_path = os.path.join(folder_path4, "GPTZero.json")
    #     # 检查data.json文件是否存在
    #     if os.path.isfile(json_file_path):
    #         # 打开并读取data.json文件
    #         with open(json_file_path, 'r') as f:
    #             data = json.load(f)
    #             for item in data:
    #                 text = item['document']
    #                 # 分词并去除停用词
    #                 tokens = text.split()
    #                 stop_words = set(stopwords.words('english'))
    #
    #                 filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #                 filtered_text = ' '.join(filtered_tokens)
    #
    #                 corpus4.append(filtered_text)
    #
    # vectorizer = TfidfVectorizer()
    # # 对文本进行向量化
    # hmtfidf_matrix = vectorizer.fit_transform(corpus4)
    # 检查是否为文件夹
    if os.path.isdir(folder_path4):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path4, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            '''进行tfidf计算'''
            with open(json_file_path, 'r') as f:
                data = json.load(f)
                for item in data:
                    text = item['document']
                    # 分词并去除停用词
                    tokens = text.split()
                    stop_words = set(stopwords.words('english'))

                    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
                    filtered_text = ' '.join(filtered_tokens)

                    singlecorpus.append(filtered_text)
                    singletfidf_matrix = vectorizer.transform(singlecorpus)
                    single_average = singletfidf_matrix.data.mean()

                    COLLEGEAIaveragelist.append(single_average)
                    singlecorpus.clear()
                    COLLEGEAIscorelist.append(item['score'])

    # '''HPA'''
    #
    # if os.path.isdir(folder_path5):
    #     # 在子文件夹中查找data.json文件
    #     json_file_path = os.path.join(folder_path5, "GPTZero.json")
    #     # 检查data.json文件是否存在
    #     if os.path.isfile(json_file_path):
    #         # 打开并读取data.json文件
    #         with open(json_file_path, 'r') as f:
    #             data = json.load(f)
    #             for item in data:
    #                 text = item['document']
    #                 # 分词并去除停用词
    #                 tokens = text.split()
    #                 stop_words = set(stopwords.words('english'))
    #
    #                 filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #                 filtered_text = ' '.join(filtered_tokens)
    #
    #                 corpus5.append(filtered_text)
    #
    # vectorizer = TfidfVectorizer()
    # # 对文本进行向量化
    # hmtfidf_matrix = vectorizer.fit_transform(corpus5)
    #
    # # 检查是否为文件夹
    # if os.path.isdir(folder_path5):
    #     # 在子文件夹中查找data.json文件
    #     json_file_path = os.path.join(folder_path5, "GPTZero.json")
    #     # 检查data.json文件是否存在
    #     if os.path.isfile(json_file_path):
    #         # 打开并读取data.json文件
    #         '''进行tfidf计算'''
    #         with open(json_file_path, 'r') as f:
    #             data = json.load(f)
    #             for item in data:
    #                 text = item['document']
    #                 # 分词并去除停用词
    #                 tokens = text.split()
    #                 stop_words = set(stopwords.words('english'))
    #
    #                 filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    #                 filtered_text = ' '.join(filtered_tokens)
    #
    #                 singlecorpus.append(filtered_text)
    #                 singletfidf_matrix = vectorizer.transform(singlecorpus)
    #                 single_average = singletfidf_matrix.data.mean()
    #
    #                 HPAaveragelist.append(single_average)
    #                 singlecorpus.clear()
    #                 HPAscorelist.append(item['perplexity'])

    plt.scatter(TOEFLaveragelist, TOEFLscorelist, label="TOFEL")
    plt.scatter(GRADE8averagelist, GRADE8scorelist, label="GRADE8")
    plt.scatter(COLLEGEaveragelist, COLLEGEscorelist, label="COLLEGE")
    plt.scatter(COLLEGEAIaveragelist, COLLEGEAIscorelist, label="COLLEGEAI")
    # plt.scatter(HPAaveragelist, HPAscorelist, label="HUMAN_PLUS_AI")

    # 添加图表标题和轴标签
    plt.title("Scatter Plot")
    plt.xlabel("Average TF-IDF")
    plt.ylabel("Score")

    # 添加图例
    plt.legend()


    plt.show()



TOEFL_folder = "./Data_and_Results/Human_Data/TOEFL_real_91"
GRADE8_folder = "./Data_and_Results/Human_Data/HewlettStudentEssay_real_88"
COLLEGE_folder = "./Data_and_Results/Human_Data/CollegeEssay_real_70"
COLLEGEAI_folder = "./Data_and_Results/GPT_Data/CollegeEssay_gpt3_31"
# HPA_folder = "./Data_and_Results/Human_Data/TOEFL_gpt4polished_91"



get_average(TOEFL_folder,GRADE8_folder,COLLEGE_folder,COLLEGEAI_folder)
