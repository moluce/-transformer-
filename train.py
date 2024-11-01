import numpy as np

from Transformer import Transformer
import random
import torch.nn as nn
import torch
from data import data_loader,dic_y


# 定义损失函数
def getloss(pred_y, y):
    # 将 pred_y 形状调整为 [b * 10, num_classes]
    pred_y = pred_y.view(-1, len(dic_y))  # 确保形状为 [b * 10, num_classes]
    return nn.CrossEntropyLoss()(pred_y, y.view(-1))  # 将 y 变为 [b * 10]
try:
    tran = torch.load('models/lastmodel.pth')
except:
    tran = Transformer()
optim = torch.optim.Adam(tran.parameters(), lr=2e-3)
sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
epochs = 1000



for epoch in range(epochs):
    all_loss = 0.0
    for i, (x, y) in enumerate(data_loader):
        #获得每个句子对的[10]
        pred_y = tran(x,y)
        # print(pred_y)
        loss = getloss(pred_y,y)
        # print('loss:',loss)
        all_loss += loss
        optim.zero_grad()
        loss.backward()
        optim.step()

    if epoch % 10 == 0:
        print(f"第{epoch}轮，损失{all_loss}")
        torch.save(tran,'models/lastmodel.pth')


        #测试准确率
        # 使用迭代器取出第一组数据
        right_count = 0
        all_count = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            # x   -> [b,10]
            # x[0] -> [10]
            predict_y = tran(x, y)
            predict_y = torch.argmax(predict_y, dim=2)
            for i in range(len(x)):  # 遍历第一个batch的所有x
                if torch.equal(y[i], predict_y[i]):
                    right_count += 1
            all_count += len(x)
        # print(all_count)
        # print(right_count)
        print(f'准确率：{right_count / all_count}')



















