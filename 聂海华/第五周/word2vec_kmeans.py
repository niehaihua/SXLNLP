#!/usr/bin/env python3
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
from collections import defaultdict

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    global average_distance, sorted_sentences
    model = load_word2vec_model(r"E:\badouai\第五周 词向量\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("E:/badouai/第五周 词向量/week5 词向量及文本向量/titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 计算类内距离
    average_distance = defaultdict(list)
    for vector_index, label in enumerate(kmeans.labels_):
        vector = vectors[vector_index]  # 句子
        center = kmeans.cluster_centers_[label]  # 对应的类内中心
        # 余弦公式计算每个句子与内类中心的距离
        distance = cosine_distance(vector, center)
        # 保存下来
        average_distance[label].append(distance)
        # 对于每一类，将类内所有文本到中心的向量余弦值取平均
    for label, distance_list in average_distance.items():
        average_distance[label] = np.mean(distance_list)
    # 按照平均距离排序，距离从小到大排序
    sorted_sentences = sorted(average_distance.items(), key=lambda x: x[1], reverse=True)

    # 按照余弦距离顺序输出
    for label, distance_avg in sorted_sentences:
        print("cluster %s , avg distance %f: " % (label, distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")


# 向量余弦距离公式
def cosine_distance(vector1, vector2):
    vector1 = np.array(vector1, dtype=np.float32)
    vector2 = np.array(vector2, dtype=np.float32)
    return 1 - np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


if __name__ == "__main__":
    main()