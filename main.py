import numpy as np
import pandas as pd
import sys
import scipy as sc
import os
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
	s=pd.read_csv(file_name,header=None)
	xt=np.array(s.values)
	xxt=[]
	for i in xt:
	    xxt.append(i[0])
	file=open("Software1.txt","w+")
	fizzbuzz=[]
	for f in xxt:
	    if (f%3==0 and f%5==0):
	        fizzbuzz.append("fizzbuzz")
	        
	    elif f%3==0:
	        fizzbuzz.append("fizz")
	        
	    elif f%5==0 :
	        
	        fizzbuzz.append("buzz")
	       
	    else:
	        
	        fizzbuzz.append(str(f))
	        
	for i in fizzbuzz:
	    file.write(str(i))
	    file.write("\n")

	#part 2
	file.close()

	model2 = torch.load("model/model2.pth")
	model2.eval()
	    
	num = 16

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
	pred = torch.max(out.data, 1)[1]#.float()


	tot += by.size(0)
	# correct += (pred == by).sum().item()
	# acc = 100*correct/tot

	j=0
	f=open("Software2.txt","w+")
	for i in pred:
	    if i==0:
	        f.write("%s\n" % "fizz")
	    elif i==1:
	        f.write("%s\n" % "buzz")
	    elif i==2:
	        f.write("%s\n" % "fizzbuzz")
	    else:
	        f.write("%s\n" % str(xxt[j]))

	    j+=1   

	f.close()
	# s1=pd.read_csv("output.txt",header=None)
	# xt1=np.array(s1.values)

	# s2=pd.read_csv("Software2.txt",header=None)
	# xt2=np.array(s2.values)

	# s3=pd.read_csv("Software1.txt",header=None)
	# xt3=np.array(s3.values)

	# a1=[]
	# a2=[]
	# a3=[]

	# for i in xt1:
	# 	a1.append(i)

	# for i in xt2:
	# 	a2.append(i)

	# for i in xt3:
	# 	a3.append(i)

	# ij=0
	# for i in range(len(a1)):
	# 	if a1[i]==a2[i]:
	# 		ij+=1
	# print("ACC:::::",int(ij/len(a1)))
	# ij=0
	# for i in range(len(a1)):
	# 	if a1[i]==a3[i]:
	# 		ij+=1
	# print("ACC:::::",int(ij/len(a1)))
	# print(len(a1),len(a2),len(a3))


        


