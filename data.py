
import random
import torch
from torch.utils.data import DataLoader

# 定义X词典
dic_X = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
dic_y = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

def get_data():
    """
    该函数返回一个tensor类型的输入句子x和输出句子y
    x的长度为：torch.Size([10])
    y的长度为：torch.Size([10])
    """

    X = []
    for i in range(10):  # 每个句子10个词
        X.append(random.choice(dic_X))

    # y是对x的变换得到的
    Y = [j.upper() for j in X][::-1]

    # 对X进行编码
    X_tensor = torch.tensor([dic_X.index(x) for x in X])

    # 对y进行编码
    y_tensor = torch.tensor([dic_y.index(y) for y in Y])
    X_tensor = X_tensor.long()
    y_tensor = y_tensor.long()


    return X_tensor, y_tensor

# 定义数据集
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()

    def __len__(self):
        return 1000

    def __getitem__(self, i):
        return get_data()



dataset = Dataset() #1000条数据 每条数据包含输入x和输出y

# 数据加载器
data_loader = DataLoader(dataset=Dataset(),
                     batch_size=128,
                     drop_last=False, #剩下的丢不丢
                     shuffle=True,
                     collate_fn=None)