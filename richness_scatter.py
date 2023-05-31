import os
import json
import matplotlib.pyplot as plt

def calculate_vocabulary_richness(text):
    words = text.split()
    unique_words = set(words)
    total_words = len(words)
    unique_word_count = len(unique_words)
    vocabulary_richness = unique_word_count / total_words
    return vocabulary_richness

def get_average(folder_path1,folder_path2,folder_path3, folder_path4):
    num = 0
    score = 0
    toeflrichness_valuelist = []
    toeflscorelist = []
    hparichness_valuelist = []
    hpascorelist = []
    collegerichness_valuelist = []
    collegescorelist = []
    collegeairichness_valuelist = []
    collegeaiscorelist = []
# 遍历根文件夹下的所有子文件夹

    # 检查是否为文件夹
    if os.path.isdir(folder_path1):
    # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path1, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            for item in data:

                vocabulary_richness = calculate_vocabulary_richness(item['document'])
                toeflrichness_valuelist.append(vocabulary_richness)
                toeflscorelist.append(item['perplexity'])



    # 检查是否为文件夹
    if os.path.isdir(folder_path2):
        # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path2, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                 data = json.load(f)
            for item in data:
                vocabulary_richness = calculate_vocabulary_richness(item['document'])
                hparichness_valuelist.append(vocabulary_richness)
                hpascorelist.append(item['perplexity'])

    if os.path.isdir(folder_path3):
    # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path3, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            for item in data:

                vocabulary_richness = calculate_vocabulary_richness(item['document'])
                collegerichness_valuelist.append(vocabulary_richness)
                collegescorelist.append(item['perplexity'])

    if os.path.isdir(folder_path4):
    # 在子文件夹中查找data.json文件
        json_file_path = os.path.join(folder_path4, "GPTZero.json")
        # 检查data.json文件是否存在
        if os.path.isfile(json_file_path):
            # 打开并读取data.json文件
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            for item in data:

                vocabulary_richness = calculate_vocabulary_richness(item['document'])
                collegeairichness_valuelist.append(vocabulary_richness)
                collegeaiscorelist.append(item['perplexity'])

    plt.scatter(hparichness_valuelist, hpascorelist, label="Human plus AI")
    plt.scatter(toeflrichness_valuelist, toeflscorelist, label="TOEFL")
    plt.scatter(collegerichness_valuelist, collegescorelist, label="College")
    plt.scatter(collegeairichness_valuelist, collegeaiscorelist, label="CollegeAI")
    plt.xlabel("Vocabulary Richness")
    plt.ylabel("Perplexity")
    plt.title("Histogram of Vocabulary Richness")
    plt.legend()
    plt.show()


HPAroot_folder = "./Data_and_Results/Human_Data/TOEFL_gpt4polished_91"

TOEFLroot_folder = "./Data_and_Results/Human_Data/TOEFL_real_91"
COLLEGEroot_folder = "./Data_and_Results/Human_Data/CollegeEssay_real_70"
COLLEGEAIroot_folder = "./Data_and_Results/GPT_Data/CollegeEssay_gpt3_31"

get_average(TOEFLroot_folder,HPAroot_folder, COLLEGEroot_folder, COLLEGEAIroot_folder)
