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

def get_average(hmfolder,gptfolder):
    num = 0
    score = 0
    hmrichness_values = []
    gptrichness_values = []
# 遍历根文件夹下的所有子文件夹
    for folder_name in os.listdir(hmfolder):
        folder_path = os.path.join(hmfolder, folder_name)

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
                    vocabulary_richness = calculate_vocabulary_richness(item['document'])
                    hmrichness_values.append(vocabulary_richness)
                    num = num + 1
                    score += vocabulary_richness

    for folder_name in os.listdir(gptfolder):
        folder_path = os.path.join(gptfolder, folder_name)

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
                    vocabulary_richness = calculate_vocabulary_richness(item['document'])
                    gptrichness_values.append(vocabulary_richness)
                    num = num + 1
                    score += vocabulary_richness

    plt.hist(hmrichness_values, bins=10, alpha=0.5, label='Human')
    plt.hist(gptrichness_values, bins=10, alpha=0.5, label='GPT')
    plt.xlabel("Vocabulary Richness")
    plt.ylabel("Frequency")
    plt.title("Histogram of Vocabulary Richness")
    plt.legend()
    plt.show()


hmroot_folder = "./Data_and_Results/Human_Data"

GPTroot_folder = "./Data_and_Results/GPT_Data"


get_average(hmroot_folder,GPTroot_folder)
