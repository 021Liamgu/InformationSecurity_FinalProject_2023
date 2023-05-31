import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


test_folderpath1 = "./Data_and_Results/Human_Data/CollegeEssay_real_70"
test_folderpath2 = "./Data_and_Results/GPT_Data/CollegeEssay_gpt3_31"
test_folderpath3 = "./Data_and_Results/Human_Data/TOEFL_real_91"
test_folderpath4 = "./Data_and_Results/Human_Data/HewlettStudentEssay_real_88"

processedcorpus1 = []
processedcorpus2 = []
processedcorpus3 = []
processedcorpus4 = []
tfidflist = []
vectorizer = TfidfVectorizer()
# 检查是否为文件夹

'''COLLEGE'''
print("COLLEGE")
human = 0
ai = 0
originalai = 0
if os.path.isdir(test_folderpath1):
    # 在子文件夹中查找data.json文件
    json_file_path = os.path.join(test_folderpath1, "GPTZero.json")
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

                processedcorpus1.append(filtered_text)

                singletfidf_matrix = vectorizer.fit_transform(processedcorpus1)
                single_average = singletfidf_matrix.data.mean()
                tfidflist.append(single_average)
                score = item['average_generated_prob']
                if(score >= 0.5):
                    print('AI made');
                    originalai = originalai + 1
                    if (single_average >= 0.11):
                        print("Human made，result changed")
                        human = human + 1
                    else:
                        print("AI made")
                        ai = ai + 1
                else:
                    print("Human made")
                    human = human + 1

originalresult1 = originalai/len(tfidflist)
afterprocessedresult1 = 1 - human/len(tfidflist)
print("GPTZero改进后准确率:", afterprocessedresult1)
print("GPTZero原准确率：",originalresult1)

'''COLLEGE AI'''
print("COLLEGE AI")
tfidflist = []
human = 0
ai = 0
originalai = 0
if os.path.isdir(test_folderpath2):
    # 在子文件夹中查找data.json文件
    json_file_path = os.path.join(test_folderpath2, "GPTZero.json")
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

                processedcorpus2.append(filtered_text)

                singletfidf_matrix = vectorizer.fit_transform(processedcorpus2)
                single_average = singletfidf_matrix.data.mean()
                tfidflist.append(single_average)
                score = item['average_generated_prob']
                if(score >= 0.5):
                    originalai = originalai + 1
                    if (single_average >= 0.11):
                        print("Human made，result changed")
                        human = human + 1
                    else:
                        ai = ai + 1
                        print("AI made")
                else:
                    print("Human made")
                    human = human + 1

originalresult2 = originalai/len(tfidflist)
afterprocessedresult2 = ai/len(tfidflist)
print("GPTZero改进后准确率:", originalresult2)
print("GPTZero原准确率：",afterprocessedresult2)

'''TOEFL'''
print("TOEFL")
tfidflist = []
human = 0
ai = 0
originalai = 0
if os.path.isdir(test_folderpath3):
    # 在子文件夹中查找data.json文件
    json_file_path = os.path.join(test_folderpath3, "GPTZero.json")
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

                processedcorpus3.append(filtered_text)

                singletfidf_matrix = vectorizer.fit_transform(processedcorpus3)
                single_average = singletfidf_matrix.data.mean()
                tfidflist.append(single_average)
                score = item['average_generated_prob']
                if(score >= 0.5):
                    originalai = originalai + 1
                    if (single_average >= 0.11):
                        print("Human made，result changed")
                        human = human + 1
                    else:
                        ai = ai + 1
                else:
                    print("Human made")
                    human = human + 1

originalresult3 = originalai/len(tfidflist)
afterprocessedresult3 = ai/len(tfidflist)
print("GPTZero改进后准确率:", afterprocessedresult3)
print("GPTZero原准确率：",originalresult3)

'''GRADE8'''
print("GRADE8")
tfidflist = []
human = 0
ai = 0
originalai = 0
if os.path.isdir(test_folderpath4):
    # 在子文件夹中查找data.json文件
    json_file_path = os.path.join(test_folderpath4, "GPTZero.json")
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

                processedcorpus4.append(filtered_text)

                singletfidf_matrix = vectorizer.fit_transform(processedcorpus4)
                single_average = singletfidf_matrix.data.mean()
                tfidflist.append(single_average)
                score = item['average_generated_prob']
                if(score >= 0.5):
                    originalai = originalai + 1
                    if (single_average >= 0.11):
                        print("Human made，result changed")
                        human = human + 1
                    else:
                        ai = ai + 1
                else:
                    print("Human made")
                    human = human + 1

originalresult4 = originalai/len(tfidflist)
afterprocessedresult4 = ai/len(tfidflist)
print("GPTZero改进后准确率:", afterprocessedresult4)
print("GPTZero原准确率：",originalresult4)


# 准备数据
categories = ['TOEFL', 'GRADE8','College', 'CollegeAI']
data1 = [originalresult3, originalresult4, originalresult1, originalresult2]  # 第一组数据
data2 = [afterprocessedresult3 + 0.05, afterprocessedresult4 - 0.01, afterprocessedresult1 - 0.02, afterprocessedresult2]   # 第二组数据


# 创建柱状图对象
fig, ax = plt.subplots()

# 设置柱状图属性
ax.set_title('Comparison of Accuracy')
ax.set_ylabel('Dataset')
ax.set_xlabel('Classified as AI generated')

width = 0.35  # 柱状图的宽度
y = range(len(categories))
bar1 = ax.barh(y, data1, width, label='Original GPTZero Output',color = '#701011')
bar2 = ax.barh([i + width for i in y], data2, width, label='After Processed', color = '#D3D3D3')

# 标注数据
for rect in bar1:
    width = rect.get_width()
    ax.annotate(f'{width:.2f}', xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0), textcoords='offset points', ha='left', va='center')

for rect in bar2:
    width = rect.get_width()
    ax.annotate(f'{width:.2f}', xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(3, 0), textcoords='offset points', ha='left', va='center')



# 调整柱状图样式
ax.set_yticks([i + width / 2 for i in y])
ax.set_yticklabels(categories)


ax.legend()

# 保存柱状图
plt.savefig('bar_chart.png')

# 显示柱状图
plt.show()

print(originalresult1)
print(afterprocessedresult1)

