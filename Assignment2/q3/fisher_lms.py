from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
from pylab import norm
from numpy import array
import pylab
import csv
    
def generateData1(file1,file2):
  inputs1=[]
  inputs2=[]
  with open(file1,'rb') as f1:
    reader = csv.reader(f1)
    for row in reader:
      inputs1.append([float(row[0]),float(row[1])])
      plt.plot(row[0],row[1],'og')

  with open(file2,'rb') as f2:
    reader = csv.reader(f2)
    for row in reader:
      inputs2.append([float(row[0]),float(row[1])])
      plt.plot(row[0],row[1],'or')
  #print inputs  
  print inputs1
  print inputs2
  return inputs1,inputs2

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
  print "Weight Vector in case of LMS--",a
  return a

if __name__ == "__main__":
  fname1=sys.argv[1]
  fname2=sys.argv[2]
  c1,c2 = generateData1(fname1,fname2)
  m1=np.mean(c1, axis=0)
  m2=np.mean(c2, axis=0)
  Sw=np.dot((c1-m1).T, (c1-m1))+np.dot((c2-m2).T, (c2-m2))
  w=np.dot(np.linalg.inv(Sw), (m2-m1))
  print "Weight vector in case of Fisher--", w
  wn=norm(w)
  wp=np.divide(w,wn)
  me=(m1+m2)/2
  b=np.dot(wp,me)
  b=np.dot(b,wp)
  linex=[]
  liney=[]
  plt.axis([-6,6,-6,6])
  for x in xrange(-6,6):
    y = (w[1]*x)/w[0]
    liney.append(y)
    linex.append(x)
  plt.plot(linex,liney,"-k",label="projection line")
  for i in c1:
    a=np.dot(wp,i)
    a=np.dot(a,wp)
    plt.plot(a[0],a[1],'og')
  for i in c2:
    a=np.dot(wp,i)
    a=np.dot(a,wp)
    plt.plot(a[0],a[1],'oy')
    line_x=[]
  line_y=[]
  for x in xrange(-6,6):
    y = b[1]-(w[0]/w[1])*(x-b[0])
    line_y.append(y)
    line_x.append(x)
  plt.plot(line_x,line_y,"--k",label="Fisher Discriminant")
  pylab.legend(loc='upper left')
  a1 = [0,0,0]
  learning_rate =0.5
  b=0.1
  class1,class2 = generateData(fname1,fname2)
  clas=[]
  clas=[each for each in class1]
  for each in class2:
    clas.append(each)
  #print clas
  a1=lms(clas,a1,learning_rate,b)
  linex=[]
  liney=[]
  for x in xrange(-10,10):
    y = float((-a1[2]-a1[0]*x)/a1[1])
    liney.append(y)
    linex.append(x)
  plt.plot(linex,liney,"g-",label="lms")
  pylab.legend(loc='upper left')
  plt.show()
  