import torch
from ultralytics.utils import imwrite

from Transformer import *
from data import data_loader,dataset

#加载已经训练好的模型
model = torch.load('models/lastmodel.pth')

# 使用迭代器取出第一组数据
right_count = 0
all_count = 0
for batch_idx, (x,y) in enumerate(data_loader):
    # x   -> [b,10]
    #x[0] -> [10]
    predict_y = model(x,y)
    predict_y = torch.argmax(predict_y, dim=2)
    for i in range(len(x)):#遍历第一个batch的所有x
        if torch.equal(y[i],predict_y[i]):
            right_count += 1
    all_count += len(x)
# print(all_count)
# print(right_count)
print(f'准确率：{right_count/all_count}')

