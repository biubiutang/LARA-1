
# coding: utf-8

# In[14]:
from nltk import FreqDist

# In[24]:

class Sentence:
    def __init__(self, wordList):
        self.wordFreqDict = FreqDist(wordList)#Dictionary of words in the sentence and corres. frequency
        self.assignedAspect = [] #list of aspects assigned to this sentence
    def __str__(self):
        return self.wordFreqDict.pformat(10000) + '##' + str(self.assignedAspect)

