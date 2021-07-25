from math import e
from torch import nn
import torch
from torch.nn import functional as F
from config import ActivityConfig as cfg

class rankpooling(nn.Module):

    def __init__(self, C, e):
        super().__init__()
        self.C = C
        self.e = e
        self.relu = nn.ReLU(inplace=True)
        self.linear = nn.Linear(2048, 1)

    ## the shape of feature map: [cfg.segment_num, 2048]
    def forward(self, features_map, _index):
        # pass
        seg_num = cfg.TRAIN.SEG_NUM
        origin_tensor = features_map.clone().detach()
        temp_tensor = torch.split(features_map, split_size_or_sections=1, dim=0)
        trans_tensor_list = []                                                  # 存放经过变换的特征

        for i in range(seg_num):
            tiny_feature = temp_tensor[i]                                       # 取特征向量的对应切片
            tiny_feature = F.normalize(tiny_feature)                            # 正则化
            tiny_feature = tiny_feature.t().mm(tiny_feature)                    # 特征自身扩展
            tiny_feature = torch.mul(tiny_feature.add(-_index[i]), -1).abs()    # 加减frame索引值
            tiny_feature = tiny_feature.add(-e)                                 # 超参数
            tiny_feature = self.relu(tiny_feature)                              # 非线性激活
            tiny_feature = self.linear(tiny_feature)                            # 维度变换
            tiny_feature = tiny_feature.t()                                     # 特征转置
            tiny_feature = tiny_feature.mul(tiny_feature)                       # 特征对应位置平方
            trans_tensor_list.append(tiny_feature)

        trans_tensor = torch.cat(trans_tensor_list, dim=0)
        final_feature = torch.add(origin_tensor.mul(self.C / 2), trans_tensor.mul((2 - self.C) / 2))

        return final_feature

if __name__ == '__main__':

    x = torch.randn([cfg.TRAIN.SEG_NUM, 2048])
    _index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    pooling = rankpooling(C=1, e=1)
    out = pooling(x, _index)
    print("shape of input: " + str(x.shape))
    print("shape of output: " + str(out.shape))
    print("input data: ")
    print(x)
    print("output data")
    print(out)