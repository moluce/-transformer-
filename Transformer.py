import math
import torch
from torch import nn
from torch.nn.functional import dropout

#定义X词典
dic_X = ['1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f','g']
dic_y = ['1','2','3','4','5','6','7','8','9','0','A','B','C','D','E','F','G']


#位置编码层
#[10,32]加上位置信息->还是[10,32]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=32):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model


    def forward(self, x):
        pe = torch.zeros(10, self.d_model)  # 初始化为0
        def get_pe(pos,i,d_model): #第pos个词 这个词的第i维 每个词有32维
            fenmu = 10000 ** (2*i/d_model)
            pe = pos / fenmu
            if i % 2 == 0:
                return math.sin(pe)
            else:
                return math.cos(pe)
        for i in range(10):  # 遍历原始数据计算位置信息 赋给位置tensor
            for j in range(self.d_model):
                pe[i, j] = get_pe(i, j, self.d_model)
        x = x + pe
        return x

#用QKV计算注意力，输入是X [10,32],输出是经过自注意力机制后的X [10,32]
def attention(Q,K,V,mask,d_model=32):
    attention = torch.matmul(Q,K.transpose(-2,-1)) #Q*K的转置
    attention = attention / d_model**0.5
    attention = attention.masked_fill_(mask, -float('inf'))
    attention = torch.softmax(attention,dim=1)
    attention = torch.matmul(attention,V)
    return attention

#自注意力层
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.layernorm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)
        self.out_fc = torch.nn.Linear(32, 32)
        self.dropout = torch.nn.Dropout(p=0.1)
    def forward(self,Q,K,V,mask):
        X_clone = Q.clone()
        Q=self.layernorm(Q)  #QKV归一化
        K = self.layernorm(K)
        V = self.layernorm(V)
        K = self.fc_K(K)  #QKV全连接运算
        V = self.fc_V(V)
        Q = self.fc_Q(Q)
        X_attention = attention(Q,K,V,mask) #计算注意力
        X_attention = dropout(self.out_fc(X_attention))#全连接
        X = X_clone + X_attention  #短接相加
        return X  #输出




class Encoderlayer(nn.Module):
    def __init__(self):
        super(Encoderlayer, self).__init__()
        self.att = Attention() #自注意力层
        self.layernorm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.out_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )

    def forward(self,X,mask):
        X = self.att(X,X,X,mask) #自注意的结果

        X_clone = X  #保存原数据
        X = self.layernorm(X)  # X归一化
        X = self.out_linear(X) #X全连接
        X = X+X_clone #与原数据相加

        return X



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer_1 = Encoderlayer()
        self.layer_2 = Encoderlayer()
        self.layer_3 = Encoderlayer()
        self.mask = torch.zeros((10, 10)).type(torch.bool)
    def forward(self,x):

        x = self.layer_1(x, self.mask)
        x = self.layer_2(x, self.mask)
        x = self.layer_3(x, self.mask)
        return x



class Decoderlayer(nn.Module):
    def __init__(self):
        super(Decoderlayer,self).__init__()
        self.att1 = Attention() #自注意力层
        self.att2 = Attention() #跨注意力层
        self.layernorm = torch.nn.LayerNorm(normalized_shape=32, elementwise_affine=True)
        self.out_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1),
        )


    def forward(self,x,y,mask_x,mask_y):
        y= self.att1(y, y, y, mask_y)
        y= self.att2(y,x,x,mask_x)
        y_clone = y.clone()
        y = self.layernorm(y)
        y = self.out_linear(y)
        y=y+y_clone
        return y



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer_1 = Decoderlayer()
        self.layer_2 = Decoderlayer()
        self.layer_3 = Decoderlayer()
        self.mask_x = torch.zeros((10, 10)).type(torch.bool)
        # 创建一个10 * 10的全零张量
        self.mask_y = torch.zeros((10, 10))
        # 通过两层循环手动设置上三角部分为1
        for i in range(1, 10):
            for j in range(10):
                if j >= i:
                    self.mask_y[i][j] = 1
        # 将mask转换为布尔型
        self.mask_y = self.mask_y.type(torch.bool)

    def forward(self, x, y):
        y = self.layer_1(x, y, self.mask_x, self.mask_y)
        y = self.layer_2(x, y, self.mask_x, self.mask_y)
        y = self.layer_3(x, y, self.mask_x, self.mask_y)
        return y




class Transformer(nn.Module):
    def __init__(self, d_model=32):
        super(Transformer, self).__init__()
        # self.emb = embed()
        self.embedding_layer = nn.Embedding(num_embeddings=17, embedding_dim=32)
        self.position = PositionalEncoding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.linear = torch.nn.Linear(32, 17)

    def forward(self, x,y):
        x = self.embedding_layer(x) #[b,10]->[b,10,32]

        x = self.position(x) #[b,10,32]
        x = self.encoder(x) #[b,10,32]

        y = self.embedding_layer(y) #[b,10,32]
        y = self.decoder(x,y)#[b,10,32]
        y = self.linear(y)#[b,10,17]
        # # 使用softmax函数将第3维度转化为概率
        # y = torch.nn.functional.softmax(y, dim=2)
        # # 获取概率最高的下标的索引，得到形状为torch.Size([10])的张量
        # y = torch.argmax(y, dim=2)

        return y



#将句子X和y编码成torch.Size([10])类型
#list->[10]意思是一句话，10个词 每个词17维
def get_encoding(X,y):
    dic_X = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'a', 'b', 'c', 'd', 'e', 'f', 'g']
    dic_y = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'A', 'B', 'C', 'D', 'E', 'F', 'G']

    # 对X进行编码
    X_tensor =torch.tensor([dic_X.index(x) for x in X],dtype=torch.long)

    # 对y进行编码
    y_tensor = torch.tensor([dic_y.index(y) for y in y], dtype=torch.long)

    return X_tensor, y_tensor

