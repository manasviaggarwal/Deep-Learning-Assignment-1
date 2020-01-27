import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.utils.data as Data
import pandas as pd
import sys
def enc(i, num):
    return np.array([i >> d & 1 for d in range(num)])


class Model(nn.Module):
    def __init__(self, numm, classes, sze):
        super(Model,self).__init__()
        layr = []   
        layr.append(nn.Linear(numm,sze))
        layr.append(nn.ReLU())
        layr.append(nn.Linear(sze,sze))
        layr.append(nn.BatchNorm1d(sze))
        layr.append(nn.ReLU())
        self.lays = nn.Sequential(*layr)
        self.outp = nn.Linear(sze, classes)

    def forward(self,x):
        x1 = self.lays(x)
        out = self.outp(x1)   
        return out 

#reading input
xt=pd.read_csv("trx.txt",header=None).values
yt=pd.read_csv("try.txt",header=None).values
xxt=[]
for i in xt:
    xxt.append(i[0])
num = 16

trX = np.array([enc(i, num) for i in xxt])

new_y=[]
for val in yt:
    if val=="Buzz":
        new_y.append(1)
    elif val=="Fizz":
        new_y.append(0)
    elif val=="FizzBuzz":
        new_y.append(2)
    else:
        new_y.append(3)
new_y=np.array(new_y)

mod = Model(16,4,150)

# print(trX[0])
dataa = Data.TensorDataset(torch.from_numpy(trX).float(),
                                    torch.from_numpy(new_y).long())

dr = Data.DataLoader(dataset=dataa,
                        batch_size=128,
                        shuffle=True)

loss1 = nn.CrossEntropyLoss()
opt = torch.optim.Adam(mod.parameters(), lr=0.001)

mod.train()

for epoch in range(1,350):
    for i,(bx, by) in enumerate(dr):
        out = mod(bx)
        loss = loss1(out, by) 
        opt.zero_grad() 
        loss.backward()
        opt.step() 
    lab = 0
    tt = 0
    pred = torch.max(out.data, 1)[1]
    tt += by.size(0)
    lab += (pred == by).sum().item()
    acc = 100*lab/tt
    print('Loss ', loss, 'train Accuracy ' ,acc)
torch.save(mod, "../model/model2.pth")
