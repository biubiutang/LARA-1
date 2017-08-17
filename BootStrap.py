
# coding: utf-8

# In[3]:

import heapq
import json
from collections import defaultdict
from ReadData import ReadData
from nltk import FreqDist
import numpy as np
modelDataDir = "modelData/"

class BootStrap:
    def __init__(self, readDataObj):
        self.corpus = readDataObj
        #Aspect,Word -> freq matrix - frequency of word in that aspect
        self.aspectWordMat = defaultdict(lambda: defaultdict(int)) 
        #Aspect --> total count of words tagged in that aspect
        # = sum of all row elements in a row in aspectWordMat matrix
        self.aspectCount = defaultdict(int)
        #Word --> frequency of jth tagged word(in all aspects) 
        # = sum of all elems in a column in aspectWordMat matrix
        self.wordCount = defaultdict(int)
        
        #Top p words from the corpus related to each aspect to update aspect keyword list
        self.p=5
        self.iter=7
        
        #List of W matrix
        self.wList=[]
        #List of ratings Dictionary belonging to review class
        self.ratingsList=[]
        #List of Review IDs
        self.reviewIdList=[]
        
        '''def calcC1_C2_C3_C4(self):
            for aspect, sentence in self.corpus.aspectSentences.items():
                for sentence in sentences:
                    for word in self.corpus.wordFreq.keys() and not in sentence.wordFreqDict.keys():
                        self.aspectNotWordMat[aspect][word]+=1
                    for word,freq in sentence.wordFreqDict.items():
                        self.aspectWordMat[aspect][word]+=freq
        '''
        
    def assignAspect(self, sentence): #assigns aspects to sentence
        sentence.assignedAspect = []
        count = defaultdict(int) #count used for aspect assignment as in paper
        #print("IN ASSIGN ASPECT FUNCTION:",len(sentence.wordFreqDict))
        for word in sentence.wordFreqDict.keys():
            for aspect, keywords in self.corpus.aspectKeywords.items():
                if word in keywords:
                    count[aspect]+=1
        if count: #if count is not empty
            maxi = max(count.values())
            for aspect, cnt in count.items():
                if cnt==maxi:
                    sentence.assignedAspect.append(aspect)
        if(len(sentence.assignedAspect)==1): #if only 1 aspect assigned to it
            self.corpus.aspectSentences[sentence.assignedAspect[0]].append(sentence)
            
    def populateAspectWordMat(self):
        self.aspectWordMat.clear()
        for aspect, sentences in self.corpus.aspectSentences.items():
            for sentence in sentences:
                for word,freq in sentence.wordFreqDict.items():
                    self.aspectWordMat[aspect][word]+=freq
                    self.aspectCount[aspect]+=freq
                    self.wordCount[word]+=freq
    
    def chiSq(self, aspect, word):
        #Total number of (tagged) word occurrences
        C = sum(self.aspectCount.values())
        
        #Frequency of word W in sentences tagged with aspect Ai
        C1 = self.aspectWordMat[aspect][word]
        
        #Frequency of word W in sentences NOT tagged with aspect Ai
        C2 = self.wordCount[word]-C1
        
        #Number of sentences of aspect A, NOT contain W
        C3 = self.aspectCount[aspect]-C1 
        
        #Number of sentences of NOT aspect A, NOT contain W
        C4 = C-C1
        
        deno = (C1+C3)*(C2+C4)*(C1+C2)*(C3+C4)
        #print(aspect, word, C, C1, C2, C3, C4)
        if deno!=0:
            return (C*(C1*C4 - C2*C3)*(C1*C4 - C2*C3))/deno
        else:
            return 0.0
        
    def calcChiSq(self):
        topPwords = {}
        for aspect in self.corpus.aspectKeywords.keys():
            topPwords[aspect] = []
        for word in self.corpus.wordFreq.keys():
            maxChi = 0.0 #max chi-sq value for this word
            maxAspect = "" #corresponding aspect
            for aspect in self.corpus.aspectKeywords.keys():
                self.aspectWordMat[aspect][word] = self.chiSq(aspect,word)
                if self.aspectWordMat[aspect][word] > maxChi:
                    maxChi = self.aspectWordMat[aspect][word]
                    maxAspect = aspect
            if maxAspect!="":
                topPwords[maxAspect].append((maxChi, word))
                
        changed=False
        for aspect in self.corpus.aspectKeywords.keys():
            for t in heapq.nlargest(self.p,topPwords[aspect]):
                if t[1] not in self.corpus.aspectKeywords[aspect]:
                    changed=True
                    self.corpus.aspectKeywords[aspect].append(t[1])
        return changed
    
    # Populate wList,ratingsList and reviewIdList
    def populateLists(self):
        for review in self.corpus.allReviews:
            #Computing W matrix for each review
            W = defaultdict(lambda: defaultdict(int))
            for sentence in review.sentences:
                if len(sentence.assignedAspect)==1:
                    for word,freq in sentence.wordFreqDict.items():
                        W[sentence.assignedAspect[0]][word]+=freq
            if len(W)!=0:
                self.wList.append(W)
                self.ratingsList.append(review.ratings)
                self.reviewIdList.append(review.reviewId)  
                
        
    def bootStrap(self):
        changed=True
        while self.iter>0 and changed:
            self.iter-=1
            self.corpus.aspectSentences.clear()
            for review in self.corpus.allReviews:
                for sentence in review.sentences:
                    self.assignAspect(sentence)
            self.populateAspectWordMat()
            changed=self.calcChiSq()
        self.corpus.aspectSentences.clear()
        for review in self.corpus.allReviews:
            for sentence in review.sentences:
                self.assignAspect(sentence)
        print(self.corpus.aspectKeywords)
    
    # Saves the object into the given file
    def saveToFile(self,fileName,obj):
        with open(modelDataDir+fileName,'w') as fp:
            json.dump(obj,fp)
            fp.close()
            
rd = ReadData()
rd.readAspectSeedWords()
rd.readStopWords()
rd.readReviewsFromJson()
rd.removeLessFreqWords()
bootstrapObj = BootStrap(rd)
bootstrapObj.bootStrap()
bootstrapObj.populateLists()
bootstrapObj.saveToFile("wList.json",bootstrapObj.wList)
bootstrapObj.saveToFile("ratingsList.json",bootstrapObj.ratingsList)
bootstrapObj.saveToFile("reviewIdList.json",bootstrapObj.reviewIdList)
bootstrapObj.saveToFile("vocab.json",list(bootstrapObj.corpus.wordFreq.keys()))
bootstrapObj.saveToFile("aspectKeywords.json",bootstrapObj.corpus.aspectKeywords)


# In[ ]:



