from sklearn import model_selection
import matplotlib.pyplot as plt_ion
import pylab
import numpy as np
from numpy import c_,r_
import matplotlib.pyplot as plt
import sys


def create_2d_array(a,b):
	return [[a,b]]

def fetchData(samples, indexes):
	data=np.empty((0,2))
	for i in indexes:
		data=np.append(data,[samples[i]],axis=0)
	return data

def predictClass_Voted(w,testsamples,c1,c2):
	correct_predictions_count=0
	for row in testsamples:
		feature_vector=row[0]
		if row[1]==c2:
			feature_vector=-1*feature_vector
		feature_vector=np.array(feature_vector)[np.newaxis]
		feature_vector=feature_vector.transpose()
		feature_vector=np.matrix(feature_vector)
		total=0	
		for tmp_wt in w:
			wt=np.matrix(tmp_wt[0])
			total = total+tmp_wt[1]*np.dot(wt,feature_vector)
		if total < 0:
			predicted_class=c2
		else:
			predicted_class=c1

		if row[1]==predicted_class:
			correct_predictions_count=correct_predictions_count+1
	l=float(len(testsamples))
	return correct_predictions_count/l


def voted_perceptron(data_classlabels,dimension,epoch):
	tmp_wt=np.zeros(dimension)
	w=np.empty((0,2))
	count=1
	for i in range(epoch):
		for row in data_classlabels:
			feature_vector=row[0]
			feature_vector=np.array(feature_vector)[np.newaxis]
			feature_vector=feature_vector.transpose()
			feature_vector=np.matrix(feature_vector)
			if np.dot(tmp_wt,feature_vector)<=0:
				w=np.append(w,create_2d_array(tmp_wt,count),axis=0)
				tmp_wt=tmp_wt+feature_vector.transpose()
				count=1
			else:
				count=count+1

	return w


def checkOtherSamplesAlso(samples,w_t):

	max_iter=10000
	iteration_count=0
	while(1):

		flag=0
		
		for row in samples:
			x=row[0]
			x=np.array(x)[np.newaxis]
			x=x.transpose()
			x=np.matrix(x)

			u=np.dot(w_t,x)

			
			if u<=0:
				# k+=1
				w_t=w_t+x.transpose()
				flag=1
				break

			iteration_count+=1

			if iteration_count>=max_iter:
				flag=0
				break

		if flag==0:
			break


	
	return w_t


def predictClass_Vanilla(weights,testData,c1,c2):
	#print weights

	#print weights.shape
	correct_predictions_count=0
	for row in testData:
		feature_vector=row[0]
		if row[1]==c2:
			feature_vector=-1*feature_vector

		feature_vector=np.array(feature_vector)[np.newaxis]
		feature_vector=feature_vector.transpose()
		feature_vector=np.matrix(feature_vector)
		wt=np.matrix(weights)
		predicted_class=np.dot(wt,feature_vector)
		if predicted_class<0:
			predicted_class=c2
		else:
			predicted_class=c1
		if row[1]==predicted_class:
			correct_predictions_count+=1

	# print "    Vanilla Perceptron accuracy: ",prediction/(float)(len(testData))

	return correct_predictions_count/(float)(len(testData))

def vanilla_perceptron(samples,dimension,epoch):
	
	w=np.zeros(dimension)
	for i in range(epoch):
		for row in samples:
			feature_vector=row[0]
			feature_vector=np.array(feature_vector)[np.newaxis]
			feature_vector=feature_vector.transpose()
			feature_vector=np.matrix(feature_vector)
			if np.dot(w,feature_vector)<=0:
				w=w+feature_vector.transpose()
	#print w.shape
	return w



def handle_Data(epoch,data_classlabels,dimension,cv,x_voted,y_voted,x_vanilla,y_vanilla,c1,c2):
		avg_accuracy_voted=0
		avg_accuracy_vanilla=0
		print "epochs: ",epoch
		for trainset, testset in cv.split(data_classlabels):
			#print trainset
			#print testset ## indexes of test data 35
			trainsamples=fetchData(data_classlabels,trainset)
			testsamples=fetchData(data_classlabels,testset)
	 		w_voted=voted_perceptron(trainsamples,dimension,epoch)
	 		w_vanilla=vanilla_perceptron(trainsamples,dimension,epoch)

			accuracy=predictClass_Voted(w_voted,testsamples,c1,c2) ## g,b
	 		avg_accuracy_voted+=accuracy
			
			accuracy=predictClass_Vanilla(w_vanilla,testsamples,c1,c2)
			avg_accuracy_vanilla+=accuracy
			#break

		avg_accuracy_voted*=10
		avg_accuracy_vanilla*=10
	 	x_voted.append(epoch)
	 	y_voted.append(avg_accuracy_voted)
	 	x_vanilla.append(epoch)
	 	y_vanilla.append(avg_accuracy_vanilla)


def Data(choice):
	print choice
	x_voted=[]
	y_voted=[]
	x_vanilla=[]
	y_vanilla=[]

	fd=open("ionosphere.data","r")
	c1='g'
	c2='b'
	if int(choice)==2:
		fd=open("breastcancer.data","r")
		c1='2'
		c2='4'
	data_classlabels=np.empty((0,2))
	for line in fd:
		line=line.strip("\n").split(",")
		feature_vector=line[:len(line)-1]
		feature_vector.append(1)
		feature_vector=np.array(feature_vector) ## Augmented feature vector
		feature_vector=feature_vector.astype(float)
		classlabel=line[len(line)-1]
		if classlabel=='b' or classlabel=='4':
			feature_vector=-1*feature_vector ## negating the feature vectors of class 'b' 
		
		data_classlabels=np.append(data_classlabels,create_2d_array(feature_vector,classlabel),axis=0)
	dimension=len(data_classlabels[0][0])
	cv = model_selection.KFold(n_splits=10)
	for epoch in epochs:
		handle_Data(epoch,data_classlabels,dimension,cv,x_voted,y_voted,x_vanilla,y_vanilla,c1,c2)
		#break
	print "Accuracies-Voted--",y_voted
	print "Accuracies-Vanilla--",y_vanilla
	plt_ion.plot(x_voted, y_voted,'bo')
	plt_ion.plot(x_vanilla, y_vanilla,'r*')
	plt_ion.plot(x_voted, y_voted,"-k",label='voted perceptron')
	plt_ion.plot(x_vanilla, y_vanilla,"--",label='vanilla perceptron')
	plt_ion.ylabel("accuracy")
	plt_ion.xlabel("epochs")
	plt_ion.legend(loc="upper left")
	plt_ion.show()

epochs=[10,15, 20, 25, 30, 35,40, 45, 50]
choice=sys.argv[1]
Data(choice)
