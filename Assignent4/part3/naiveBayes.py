import pandas as pd
import numpy as np
import math 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

data = pd.read_csv("census-income.data")
categorical_attributes = ['ACLSWKR','ADTIND','ADTOCC','AHGA','AHSCOL','AMARITL','AMJIND','AMJOCC','ARACE',
                          'AREORGN','ASEX','AUNMEM','AUNTYPE','AWKSTAT','FILESTAT','GRINREG','GRINST','HHDFMX',
                          'HHDREL','MIGMTR1','MIGMTR3','MIGMTR4','MIGSAME','MIGSUN','PARENT','PEFNTVTY',
                          'PEMNTVTY','PENATVTY','PRCITSHP','SEOTR','VETQVA','VETYN','YEAR']
continuous_attributes = ['AAGE','AHRSPAY','CAPGAIN','CAPLOSS','WKSWORK']
cl=['INCOME']
#Handling missing values
data = data.loc[:,categorical_attributes+continuous_attributes+cl]
mode = data.loc[data.GRINST != " ?", "GRINST"].mode().iloc[0]
data.loc[data.GRINST == " ?", "GRINST"] = mode

mode = data.loc[data.MIGMTR3 != " ?", "MIGMTR3"].mode().iloc[0]
data.loc[data.MIGMTR3 == " ?", "MIGMTR3"] = mode

mode = data.loc[data.MIGMTR4 != " ?", "MIGMTR4"].mode().iloc[0]
data.loc[data.MIGMTR4 == " ?", "MIGMTR4"] = mode

mode = data.loc[data.MIGSAME != " ?", "MIGSAME"].mode().iloc[0]
data.loc[data.MIGSAME == " ?", "MIGSAME"] = mode

mode = data.loc[data.PEFNTVTY != " ?", "PEFNTVTY"].mode().iloc[0]
data.loc[data.PEFNTVTY == " ?", "PEFNTVTY"] = mode

mode = data.loc[data.PEMNTVTY != " ?", "PEMNTVTY"].mode().iloc[0]
data.loc[data.PEMNTVTY == " ?", "PEMNTVTY"] = mode

mode = data.loc[data.PENATVTY != " ?", "PENATVTY"].mode().iloc[0]
data.loc[data.PENATVTY == " ?", "PENATVTY"] = mode

data["MIGMTR1"] = str(data["MIGMTR1"])

for attribute in continuous_attributes:
	print(attribute,np.mean(data.loc[:,attribute]),np.std(data.loc[:,attribute]))
#print("Mean -",means)
#print("Standard Deviation -",std_dev)
input()

#Binning some of the Continuous Attributes
data["WAGE_BIN"] = pd.cut(data.AHRSPAY, bins=1240, labels=False)
categorical_attributes.insert(0, "AHRSPAY_BIN")

data["CAPGAIN_BIN"] = pd.cut(data.CAPGAIN, bins=132, labels=False)
categorical_attributes.insert(0, "CAPGAIN_BIN")

data["CAPLOSS_BIN"] = pd.cut(data.CAPLOSS, bins=113, labels=False)
categorical_attributes.insert(0, "CAPLOSS_BIN")
#means=[]
#std_dev=[]

Accuracy = np.array([],dtype=float)
for i in range(30):
	kf = KFold(n_splits=10,shuffle=True)
	data = np.array(data)
	accuracy = 0.0
	for train_index, test_index in kf.split(data):
		allAttributes=categorical_attributes+cl
		trainData = pd.DataFrame(data[train_index], columns=allAttributes)
		testData = pd.DataFrame(data[test_index], columns=allAttributes)
		classLessIncome = trainData.loc[trainData.INCOME == " - 50000.", :]
		classMoreIncome = trainData.loc[trainData.INCOME == " 50000+.", :]
		classLessMean = {}
		classMoreMean = {}
		classLessStd = {}
		classMoreStd = {}
		numLess = classLessIncome.shape[0]
		numMore = classMoreIncome.shape[0]
		probLess = math.log(numLess/trainData.shape[0])  #a-priori probability (log probabilities)
		probMore = math.log(numMore/trainData.shape[0])
		classLessDict = {}
		classMoreDict = {}
		for attribute in categorical_attributes:
			classLessDict[attribute] = dict(classLessIncome[attribute].value_counts()/numLess)
			classMoreDict[attribute] = dict(classMoreIncome[attribute].value_counts()/numMore)
		#testing phase
		tmpAcc = 0.0
		for i in range(testData.shape[0]):
			X = testData.iloc[i:i+1]
			posteriorLess = probLess
			posteriorMore = probMore
			for attribute in categorical_attributes[:-1]:
				if X[attribute].iloc[0] in classLessDict[attribute].keys():
					posteriorLess += math.log(classLessDict[attribute][X[attribute].iloc[0]])
				if X[attribute].iloc[0] in classMoreDict[attribute].keys():
					posteriorMore += math.log(classMoreDict[attribute][X[attribute].iloc[0]])
			if posteriorLess > posteriorMore and X['INCOME'].iloc[0] == " - 50000.":
				tmpAcc += 1
			elif posteriorMore > posteriorLess and X['INCOME'].iloc[0] == " 50000+.": 
				tmpAcc += 1
		tmpAcc /= testData.shape[0]
		accuracy += tmpAcc
	accuracy /= 10
	print(accuracy)
	Accuracy = np.append(Accuracy, accuracy)
print("Mean: ", np.mean(Accuracy))
print("Std-Dev: ", np.std(Accuracy))
