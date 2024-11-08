# coding:utf8
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

"""
基于pytorch的Bert语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        # 使用bert模型
        self.bert = BertModel.from_pretrained(pretrain_model_path, return_dict=False)
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if len(mask.shape) != 2:
            raise ValueError(f"Attention mask should be a 2D tensor, but got shape {mask.shape}")
        if y is not None:
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            x, _ = self.bert(x)
            y_pred = self.classify(x)
            return torch.softmax(y_pred, dim=-1)


# 加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus


# 随机生成一个样本
# 从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        # cls_token_id开始，sep_token_id分隔
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [
            tokenizer.sep_token_id]
        # 创建一个长度为prompt_encode的列表，值都是-1.这里用【-1】分隔
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        # 使用mask矩阵，让prompt_encode可以互交，answer_encode没有
        mask = create_mask(len(prompt_encode), len(answer_encode))
        # 如果样本长度超过最大长度，则截断,变成0
        x = x[0:max_length] + (max_length-len(x)) * [0]
        y = y[0:max_length] + (max_length-len(y)) * [0]
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


# 创建掩码矩阵，主要是让token之间不可见
def create_mask(s1, s2):
    len_s1 = s1 + 2  # cls和sep
    len_s2 = s2 + 1  # sep
    # 创建掩码张量
    mask = torch.ones((len_s1 + len_s2, len_s1 + len_s2))
    # s1的当前token不能看到s2的任何token
    for i in range(len_s1):
        mask[0, i, len_s1:] = 0  # 不可见的都为0
    for i in range(len_s2):
        # s2的当前token不能看到后面的s2的token
        mask[0, len_s1 + i, len_s1 + 1 + 1:] = 0
    return mask


def pad_mask(tensor, tensor_shape):
    # 获取输入张量的长和宽
    height, width = tensor.shape
    # 目标张量的长和宽
    target_height, target_width = tensor_shape
    # 创建一个全零的张量，形状为目标张量。数据类型与输入张量tensor相同，并且设备与输入张量相同
    result = torch.zeros(tensor_shape, dtype=tensor.dtype, device=tensor.device)
    # 进行截断
    start_height = 0
    end_height = min(height, target_height)
    start_width = 0
    end_width = min(width, target_width)
    # 将输入张量的数据复制到目标张量中
    result[start_height:end_height, start_width:end_width] = tensor[:end_height-start_height, :end_width-start_width]
    return result


# 建立模型
def build_model(vocab_size, hidden_size, pretrain_model_path):
    model = LanguageModel(hidden_size, vocab_size, pretrain_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        # 生成了换行符，或生成文本超过100字则终止迭代
        while pred_char != "\n" and len(openings) <= 100:
            openings += pred_char
            x = tokenizer.encode(openings, add_special_tokens=False)
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = ''.join(tokenizer.decode(index))
    return openings


def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


def train(corpus_path, save_weight=True):
    epoch_num = 200  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    max_length = 50  # 样本文本长度
    char_dim = 768  # 每个字的维度
    vocab_size = 21128  # 词表大小
    learning_rate = 0.001  # 学习率

    pretrain_model_path = r"E:/badouai/bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)  # 加载语料
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  # 建立数据集
    model = build_model(vocab_size, char_dim, pretrain_model_path)  # 建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data:  # 构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()  # 梯度归零
            loss = model(x, mask, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("姑娘钱包跌落西湖 小伙冒寒入水捞回", model, tokenizer))
        print(generate_sentence("北美洲发现肥皂人", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    # build_vocab_from_corpus("corpus/all.txt")
    train('sample_data.json', False)