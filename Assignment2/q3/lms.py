from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import csv
from pylab import norm
from numpy import array
    
def generateData(file1,file2):
  inputs1=[]
  inputs2=[]
  with open(file1,'rb') as f1:
    reader = csv.reader(f1)
    for row in reader:
      inputs1.append([float(row[0]),float(row[1]),1])
      plt.plot(row[0],row[1],'ob')

  with open(file2,'rb') as f2:
    reader = csv.reader(f2)
    for row in reader:
      inputs2.append([-1*float(row[0]),-1*float(row[1]),-1])
      plt.plot(row[0],row[1],'or')
  #print inputs  
  return inputs1,inputs2


def lms(data,a,learning_rate,b):
	e=0.0001
	k=1
	n=len(data)
	a+= learning_rate * (b-np.dot(a,data[0]))*array(data[0])/k
	step=0
	while step <1000:
		if k==0:
			k=1
		a+= learning_rate * (b-np.dot(a,data[k]))*array(data[k])/k
		if norm(b-np.dot(a,data[k])) < e:
			break
		step+=1
		k=(k+1)%n
	print a
	return a
         
def main():
    fname1=sys.argv[1]
  fname2=sys.argv[2]
  a = [0,0,0]
  learning_rate =0.5
  b=0.1
  class1,class2 = generateData(fname1,fname2)
  clas=[]
  clas=[each for each in class1]
  for each in class2:
  	clas.append(each)
  #print clas
  a=lms(clas,a,learning_rate,b)
  linex=[]
  liney=[]
  for x in xrange(-10,10):
	y = float((-a[2]-a[0]*x)/a[1])
	liney.append(y)
	linex.append(x)
  plt.plot(linex,liney,"-k")
  plt.show()

   
main()