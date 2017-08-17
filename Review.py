
# coding: utf-8

# In[25]:
from collections import defaultdict

class Review:
    def __init__(self):
        self.sentences = [] #list of objects of class Sentence
        self.reviewId = ""
        self.ratings = {} #true ratings provided by the user
        
    def __str__(self):
        retStr = ""
        for sentence in self.sentences:
            retStr += sentence.__str__() + '\n'
        retStr += "###"+self.reviewId+"###"+str(self.ratings)+"\n"
        return retStr
    
