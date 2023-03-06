from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch import optim
import os.path
import numpy as np
import json
import pandas as pd


class MultiLinearDataset(Dataset):
    def __init__(self, filename):
        super(MultiLinearDataset, self).__init__()
        df = pd.read_csv(filename)
        df = df.drop(['Id'], axis=1)
        self.species = df['Species']
        self.data = df.drop(['Species'], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor([self.data.iloc[idx]]), \
               self.species.iloc[idx]

    def getSpeciesList(self):
        return pd.unique(self.species).tolist()

    def getDataDim(self):
        return len(self.data.iloc[0])

    def generateLinearData(self, speciesList):
        species = []
        data = []
        for idx in range(len(self.species)):
            if self.species.iloc[idx] in speciesList:
                species.append(self.species.iloc[idx])
                data.append(self.data.iloc[idx])
        return species, data



class LinearDataset(Dataset, ABC):
    def __init__(self, data, species):
        super(LinearDataset, self).__init__()
        self.species = species
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.Tensor(self.data[idx]), \
               self.species[idx]


# 单分类器
class LinearClassifier(nn.Module):
    def __init__(self,
                 speciesList=[], #类型列表
                 dim=None, #维度
                 ):
        super(LinearClassifier, self).__init__()
        self.speciesList = speciesList
        self.dim = dim
        self.speNum = 0 #类型总数
        self.speciesToNumTable = {} #建立类名到数字的映射
        for spec in self.speciesList:
            if spec not in self.speciesToNumTable.keys():
                self.speciesToNumTable[spec] = self.speNum
                self.speNum += 1
        Theta = torch.randn([1, dim]) #[1, dim]权重数组
        bias = torch.randn([1, 1]) #[1, 1] 偏执大小
        self.Theta = nn.Parameter(Theta) #转化成可训练参数
        self.bias = nn.Parameter(bias) #转化成可训练参数
        self.activation = torch.sigmoid #激活函数


    def forward(self, inputData):
        batch_size = inputData.shape[0]
        Theta = torch.repeat_interleave(self.Theta.unsqueeze(0), repeats=batch_size, dim=0)  # [batch_size, 1, dim]
        # 指数
        mul = torch.bmm(Theta, inputData.transpose(1, 2)).squeeze()  # [batch_size]
        b = torch.repeat_interleave(self.bias, repeats=batch_size, dim=0).squeeze()  # [batch_size]
        res = mul + b  # [batch_size]
        cls = self.activation(res) #激活函数
        return cls


def Loss(opt, tgt):
    num = opt.shape[0]
    loss = 0
    for idx in range(num):
        loss += tgt[idx] * torch.log(opt[idx]) + (1-tgt[idx]) * torch.log(1-opt[idx])
    return -loss/num


def Train(dataset, #数据集
          model, #模型
          epochs=500, #轮数
          batch_size=10, #batch大小
          rate=1e-4, #学习率
          gdFunc=optim.SGD, #梯度下降函数
          lossFunc=Loss, #损失函数
          ):
    dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = gdFunc(model.parameters(), lr=rate)
    for e in range(epochs):
        for idx, batch in enumerate(dataLoader):
            ipt, tgt = batch #获取训练数据
            ipt = ipt.unsqueeze(0)
            optimizer.zero_grad()
            opt = model(ipt)
            numTgt = []
            for s in tgt:
                numTgt.append([model.speciesToNumTable[s]])
            loss = lossFunc(opt, torch.Tensor(numTgt))
            loss.backward()
            optimizer.step()
        if e%10 == 0:
            print('Epoch:', '%04d' % e, 'cost =', '{:.6f}'.format(float(loss)))


# 多分类器
class MultiLinearClassifier(nn.Module):
    def __init__(self,
                 speciesList=[], #物种列表
                 dim=None, #数据维度
                 ):
        super(MultiLinearClassifier, self).__init__()
        self.speciesList = speciesList
        self.speciesNum = len(speciesList)
        self.ClassifierList = []
        for i in range(len(speciesList)):
            for j in range(i+1, len(speciesList)):
                spl = [speciesList[i], speciesList[j]]
                linearClassifier = LinearClassifier(speciesList=spl, dim=dim)
                self.ClassifierList.append(linearClassifier)

    def forward(self, inputData):
        speciesCount = {}
        for species in self.speciesList:
            if species not in speciesCount.keys():
                speciesCount[species] = 0
        for classifier in self.ClassifierList:
            res = classifier(inputData)
            speciesCount[classifier.speciesList[0]] += (1 - res)
            speciesCount[classifier.speciesList[1]] += res
        maxProb = 0
        resSpecies = self.speciesList[0]
        for species in self.speciesList:
            if speciesCount[species] > maxProb:
                maxProb = speciesCount[species]
                resSpecies = species
        return resSpecies

    def MultiTrain(self,
                   multiData,
                   epochs=10000,  # 轮数
                   batch_size=20,  # batch大小
                   rate=1e-4,  # 学习率
                   gdFunc=optim.SGD,  # 梯度下降函数
                   lossFunc=Loss,  # 损失函数
                   ):
        idx = 0
        for i in range(len(self.speciesList)):
            for j in range(i+1, len(self.speciesList)):
                species, data = multiData.generateLinearData([self.speciesList[i], self.speciesList[j]])
                dataset = LinearDataset(species=species, data=data)
                Train(dataset=dataset, model=self.ClassifierList[idx], epochs=epochs, batch_size=batch_size,
                      rate=rate, gdFunc=gdFunc, lossFunc=lossFunc)
                idx += 1


def Test():
    pass


if __name__ == "__main__":
    filename_ = "Iris.csv"
    savename = "model1.pth"
    MultilinearDataset = MultiLinearDataset(filename_)
    if not os.path.exists(savename):
        Model = MultiLinearClassifier(speciesList=MultilinearDataset.getSpeciesList()
                                  , dim=MultilinearDataset.getDataDim())
        Model.MultiTrain(MultilinearDataset)
        torch.save(Model, savename)
    else:
        Model = torch.load(savename)
        testLoader = DataLoader(MultilinearDataset, batch_size=1)
        for index, Batch in enumerate(testLoader):
            Input, Target = Batch
            print(Model(Input), Target)




