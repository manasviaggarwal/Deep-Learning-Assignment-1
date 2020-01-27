import numpy as np
import pandas as pd
import sys
import scipy as sc
import os
import re
import itertools
import statistics
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torch.utils.data as Data
import pandas as pd
import sys
print("NAME:  MANASVI AGGARWAL")
print("DEPARTMENT:  CSA")
print("COURSE:  MTECH(RES.)")
print("SR NO:  16223")

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
#part 1
if __name__ == "__main__":
	
	file_name = sys.argv[2]
	s=pd.read_csv(file_name)
	xt=np.array(s.values)
	xxt=[]
	for i in xt:
	    xxt.append(i[0])
	file=open("software1.txt","w+")
	fizzbuzz=[]
	for f in xxt:
	    if (f%3==0 and f%5==0):
	        fizzbuzz.append("FizzBuzz")
	        
	    elif f%3==0:
	        fizzbuzz.append("Fizz")
	        
	    elif f%5==0 :
	        
	        fizzbuzz.append("Buzz")
	       
	    else:
	        
	        fizzbuzz.append(str(f))
	        
	for i in fizzbuzz:
	    file.write(str(i))
	    file.write("\n")

	#part 2

	#reading input
	#test phase
	#reading input
	# data = str(sys.argv[2])
	# xt=pd.read_csv(data).values


	model2 = torch.load("model/model2.pth")
	model2.eval()

	# xxt=[]
	# for i in xt:
	# #     print(i[0])   
	#     xxt.append(i[0])
	    
	num = 16

	# xxt=np.array(xxt)
	# print(xxt)
	trX = np.array([enc(i, num) for i in xxt])

	new_y=[]
	for val in xt:
	    if (val%3==0 and val%5==0):
	        new_y.append(2)
	    elif val%3==0:
	        new_y.append(0)
	    elif val%5==0 :
	        new_y.append(1)
	    else:
	        new_y.append(3)

	new_y=np.array(new_y)
	bx=torch.from_numpy(trX).float()
	by=torch.from_numpy(new_y).float()

	out = model2(bx)
	correct = 0
	tot = 0
	pred = torch.max(out.data, 1)[1]


	tot += by.size(0)
	correct += (pred == by).sum().item()
	acc = 100*correct/tot
	# print('Test Accuracy',acc)
	j=0
	f=open("software2.txt","w+")
	for i in pred:
	    if i==0:
	        f.write("%s\n" % "Fizz")
	    elif i==1:
	        f.write("%s\n" % "Buzz")
	    elif i==2:
	        f.write("%s\n" % "FizzBuzz")
	    else:
	        f.write("%s\n" % str(xxt[j]))

	    j+=1   

	f.close()


