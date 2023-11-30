import re
import jieba
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim import corpora, models
from gensim.matutils import sparse2full
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models import word2vec
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras import Input
from keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPool1D, Embedding
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences




# 全角转半角
def full_to_half(text:str):      # 输入为一个句子
    _text = ""
    for char in text:
        inside_code = ord(char)     # 以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值
        if inside_code == 12288:    # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        _text += chr(inside_code)
    return _text

# 文本清洗，过滤非文本内容
def clear_character(sentence):
    #pattern = re.compile("[^\u4e00-\u9fa5^,^.^!^a-z^A-Z^0-9]")  # 只保留中英文、数字和符号，去掉其他东西
    pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")         # 只保留中英文和数字
    line = re.sub(pattern, '', sentence)  # 把文本中匹配到的字符替换成空字符
    new_sentence = ''.join(line.split())  # 去除空白
    return new_sentence

# 分词函数
def cut_word(text):
    text = jieba.cut(text)
    # for i in text:
    #     print(i)
    return text

# 停用词函数
def drop_stopwords(contents, stopwords):
    contents_clean = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
        contents_clean.append(line_clean)
    return contents_clean



def main():
    # 获取要处理的数据
    with open('info.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()
            # print(data[5736])

    # 全角转半角
    for i in range(0, len(data)):
        data[i] = full_to_half(data[i])
    # print(data)

    # 过滤非文本
    for i in range(0, len(data)):
        data[i] = clear_character(data[i])
    # print(data)

    # 创建数组
    fc_data = []

    # 分词
    for i in range(0, len(data)):
        zjfc_data = []
        text = cut_word(data[i])
        for j in text:
            zjfc_data.append(j)
        fc_data.append(zjfc_data)
    # print(fc_data)

    # 停用词表
    with open('cn_stopwords.txt', 'r', encoding='utf-8') as f:
            stopwords = f.readlines()
    for i in range(0, len(stopwords)):
        stopwords[i] = full_to_half(stopwords[i])
    for i in range(0, len(stopwords)):
        stopwords[i] = clear_character(stopwords[i])

    # 去掉文本中的停用词
    qu_stopword_data = drop_stopwords(fc_data, stopwords)
    #print(qu_stopword_data)

    # 去除空数组
    finall_data = list(filter(None, qu_stopword_data))
    # print(finall_data)

    # 构建词袋模型
    dictionary = corpora.Dictionary(finall_data)
    corpus = [dictionary.doc2bow(text) for text in finall_data]
    # print(dictionary.token2id) # 得到每个词编号

    # 标签
    csv_reader = csv.reader(open("lda_topics_7.csv")) # csv_reader = csv.reader(open("lda_topics.csv"))
    i = 0
    for row in csv_reader:
        if i % 2 == 0:
            labels = row
    # print(labels)
    new_labels = []
    for label in labels:
        new_labels.append(int(label))
    print(new_labels)

    # fit_on_texts函数可以将输入的文本每个词编号 编号根据词频(词频越大编号越小)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(finall_data)
    vocab = tokenizer.word_index  # 停用词已过滤,获取每个词的编号
    print(vocab)
    # 使用 train_test_split 分割 X y 列表
    X_train, X_test, y_train, y_test = train_test_split(finall_data, new_labels, test_size=0.3, random_state=1)
    print(X_train[:5])
    print(y_train[:5])

    # ----------------------------------第三步 词向量构建--------------------------------
    # Word2Vec训练
    maxLen = 100  # 词序列最大长度
    num_features = 100  # 设置词语向量维度
    min_word_count = 3  # 保证被考虑词语的最低频度
    num_workers = 4  # 设置并行化训练使用CPU计算核心数量
    context = 4  # 设置词语上下文窗口大小

    # 设置模型
    model = word2vec.Word2Vec(finall_data, workers=num_workers, vector_size=num_features,
                              min_count=min_word_count, window=context)
    # 强制单位归一化
    model.init_sims(replace=True)
    # 输入一个路径保存训练模型 其中./data/model目录事先存在
    model.save("CNNw2vModel")
    model.wv.save_word2vec_format("CNNVector", binary=False)
    print(model)
    # 加载模型 如果word2vec已训练好直接用下面语句
    w2v_model = word2vec.Word2Vec.load("CNNw2vModel")

    # 特征编号(不足的前面补0)
    trainID = tokenizer.texts_to_sequences(X_train)
    print(trainID)
    testID = tokenizer.texts_to_sequences(X_test)
    print(testID)
    # 该方法会让CNN训练的长度统一
    trainSeq = pad_sequences(trainID, maxlen=maxLen)
    print(trainSeq)
    testSeq = pad_sequences(testID, maxlen=maxLen)
    print(testSeq)

    # 标签独热编码 转换为one-hot编码
    trainCate = to_categorical(y_train, num_classes=7) # trainCate = to_categorical(y_train, num_classes=2)
    print(trainCate)
    testCate = to_categorical(y_test, num_classes=7)  # testCate = to_categorical(y_test, num_classes=2)
    print(testCate)

    # ----------------------------------第四步 CNN构建--------------------------------
    # 利用训练后的Word2vec自定义Embedding的训练矩阵 每行代表一个词(结合独热编码和矩阵乘法理解)
    embedding_matrix = np.zeros((len(vocab) + 1, 100))  # 从0开始计数 加1对应之前特征词
    for word, i in vocab.items():
        try:
            # 提取词向量并放置训练矩阵
            embedding_vector = w2v_model.wv[str(word)]
            embedding_matrix[i] = embedding_vector
        except KeyError:  # 单词未找到跳过
            continue

    # 训练模型
    main_input = Input(shape=(maxLen,), dtype='float64') # 初始化深度学习网络输入层的tensor(一个多维数组) shape表示输入将是一个maxLen维的向量
    # 词嵌入 使用预训练Word2Vec的词向量 自定义权重矩阵 100是输出词向量维度 # 由于传统模型(词袋模型)产生的词向量大而稀疏，词嵌入是对传统模型的改进
    embedder = Embedding(len(vocab) + 1, 100, input_length=maxLen,
                         weights=[embedding_matrix], trainable=False)  # 不再训练
    # 建立模型
    model_cnn = Sequential()
    model_cnn.add(embedder)  # 构建Embedding层
    model_cnn.add(Conv1D(256, 3, padding='same', activation='relu'))  # 卷积层步幅3
    model_cnn.add(MaxPool1D(maxLen - 5, 3, padding='same'))  # 池化层
    model_cnn.add(Conv1D(32, 3, padding='same', activation='relu'))  # 卷积层
    model_cnn.add(Flatten())  # 拉直化
    model_cnn.add(Dropout(0.3))  # 防止过拟合 30%不训练
    model_cnn.add(Dense(256, activation='relu'))  # 全连接层
    model_cnn.add(Dropout(0.2))  # 防止过拟合
    model_cnn.add(Dense(units=7, activation='softmax'))  # 输出层 model_cnn.add(Dense(units=2, activation='softmax'))

    # 模型可视化
    model_cnn.summary()

    # 激活神经网络
    model_cnn.compile(optimizer='adam',  # 优化器
                  loss='categorical_crossentropy',  # 损失
                  metrics=['accuracy']  # 计算误差或准确率
                  )

    # 训练(训练数据、训练类标、batch—size每次256条训练、epochs、随机选择、验证集20%)
    history = model_cnn.fit(trainSeq, trainCate, epochs=6, batch_size=256, validation_split=0.2)
    model_cnn.save("TextCNN")

    # ----------------------------------第五步 预测模型--------------------------------
    # 预测与评估
    mainModel = load_model("TextCNN")
    result = mainModel.predict(testSeq)  # 测试样本
    print(result)
    print(np.argmax(result, axis=1))
    score = mainModel.evaluate(testSeq,
                               testCate,
                               batch_size=32)
    print(score)

if __name__ == '__main__':
    main()