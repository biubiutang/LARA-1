import numpy
import matplotlib.pyplot as plt
import pylab
import sys
import csv

def read_data(file1,file2,x):
	data=[]
	data1=[]
	xs=[]
	ys=[]
	with open(file1,'rb') as f1:
		reader = csv.reader(f1)
		for row in reader:
			data.append([float(row[0]),float(row[1]),1])
			data1.append([float(row[0]),float(row[1]),1])
			xs.append(float(row[0]))
			ys.append(float(row[1]))
	plt.scatter(xs,ys,color='red',label='voted perceptron')
	xs1=[]
	ys1=[]
	with open(file2,'rb') as f2:
		reader = csv.reader(f2)
		for row in reader:
			data1.append([float(row[0]),float(row[1]),1])
			data.append([-1*float(row[0]),-1*float(row[1]),-1])
			xs1.append(float(row[0]))
			ys1.append(float(row[1]))
	plt.scatter(xs1,ys1,color='blue')
	xmin = min(float(s[0]) for s in data1)-1
	xmax = max(float(s[0]) for s in data1)-1
	x.append(xmin)
	x.append(xmax)
	return data

	plt.plot(xs,ys,'ob',label='class1')
	plt.plot(xs1,ys1,'or',label='class2')
	pylab.legend(loc='upper left')

def online_perceptron(data,a,learning_rate):
	k=0
	flag = False
	while flg == False:
		k=k+1
		flg = True
		for i in xrange(len(data)):
			temp = numpy.inner(a,data[i])
			if temp <= 0 :
				delta = numpy.dot(data[i],learning_rate)
				a = numpy.add(a,delta)
				flg = False
	print "iterations = "+str(k)
	return a

if __name__ == "__main__":
	a = [0,0,0]
	learning_rate =0.1
	fname1=sys.argv[1]
	fname2=sys.argv[2]
	data=[]
	x=[]
	data=read_data(fname1,fname2,x)
	a = online_perceptron(data,a,learning_rate)
	print a
	y = [ (-1*a[0]*each - a[2])/a[1] for each in x]
	plt.plot(x,y)
	plt.show()

