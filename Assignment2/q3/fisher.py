from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
from pylab import norm
import pylab
import csv
    
def generateData(file1,file2):
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



if __name__ == "__main__":
	fname1=sys.argv[1]
	fname2=sys.argv[2]
	c1,c2 = generateData(fname1,fname2)
	m1=np.mean(c1, axis=0)
	m2=np.mean(c2, axis=0)
	Sw=np.dot((c1-m1).T, (c1-m1))+np.dot((c2-m2).T, (c2-m2))
	w=np.dot(np.linalg.inv(Sw), (m2-m1))
	print "weight vector--", w
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
		plt.plot(a[0],a[1],'ob')
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
	
  	plt.show()
	