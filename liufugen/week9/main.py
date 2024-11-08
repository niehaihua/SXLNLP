# -*- coding: utf-8 -*-
import time

import pandas as pd
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data


logger = logging.Logger


def init():
    # [DEBUG, INFO, WARNING, ERROR, CRITICAL]
    # 1、设置全局的日志格式和级别
    global logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # 2、获取logger （给日志器起个名字 "__name__"）
    logger = logging.getLogger(__name__)  # __name__内置变量模块名称，轻松地识别出哪个模块产生了哪些日志消息（主程序模块）

    # 3、创建文件处理器，指定日志文件和日志级别（局部）---文件输出FileHandle（输出到指定文件）
    file_handler = logging.FileHandler('output/application.log')  # 指定日志文件名application.log，默认在当前目录下创建
    file_handler.setLevel(logging.INFO)  # 设置日志级别(只输出对应级别INFO的日志信息)
    # 设置日志格式
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%m/%d/%Y %H:%M:%S'))

    # 4、添加文件处理器到logger
    logger.addHandler(file_handler)

"""
模型训练主程序
"""

def main(config,category_index):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_id, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        acc = evaluator.eval(epoch)

    model_path = os.path.join(config["model_path"], f"第_{category_index}种.pth")
    torch.save(model.state_dict(), model_path)  # 保存模型权重
    return acc


if __name__ == "__main__":
    init()
    # main(Config,1)
    lrs = [1e-3, 1e-4, 1e-5]
    batch_sizes = [16,32,64]
    epochs = [10, 15, 20]
    optimizers = ["adam", "sgd"]


    # #对比所有模型
    header = ["类别序号"]
    header.extend(
        ["层数", "训练轮数", "批次大小",
         "优化器", "学习率", "运行时间", "最后一轮准确率"])
    data_df = pd.DataFrame(columns=header)
    data_df.to_csv("output/category.csv", mode="a", index=False, encoding="utf-8")
    category_index = 1

    for lr in lrs:
        Config["learning_rate"] = lr
        for batch_size in batch_sizes:
            Config["batch_size"] = batch_size
            for epoch in epochs:
                Config["epoch"] = epoch
                for optimizer in optimizers:
                    Config["optimizer"] = optimizer
                    logger.info(f"----------------------------第{category_index}种类型-------------------------")
                    start = time.time()
                    accuracy = main(Config,category_index)
                    end = time.time()

                    result = [f"第{category_index}种"]
                    result.extend([Config["num_layers"],
                                   Config["epoch"], Config["batch_size"],
                                   Config["optimizer"], Config["learning_rate"], end - start, accuracy])
                    data_df.loc[0] = result
                    data_df.to_csv("output/category.csv", mode="a", index=False, header=False,
                                   encoding="utf-8")

                    logger.info(
                        f"最后一轮准确率：{accuracy}, 当前配置：{Config},耗时：{end - start} \n\n\n")
                    category_index += 1
