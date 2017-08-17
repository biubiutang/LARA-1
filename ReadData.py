
# coding: utf-8

# In[2]:

import json
import nltk
from nltk.corpus import stopwords
import os 
import glob
from nltk.stem.porter import *
from nltk import FreqDist
path="hotelReviews/"
projectSettings="settings/"
from collections import defaultdict
import string
from Sentence import Sentence
from Review import Review

class ReadData:
    def __init__(self):
        self.aspectKeywords = {} #aspect name <--> keywords list
        self.stopWords = []
        self.wordFreq = {} #dict with of all words and their freq in the corpus
        self.lessFrequentWords=set() #words which have frequency<5 in the corpus
        self.allReviews = [] #list of Review objects from the whole corpus
        self.aspectSentences = defaultdict(list) #aspect to Sentences mapping
        self.wordIndexMapping={} #word to its index in the corpus mapping
        self.aspectIndexMapping={} #aspect to its index in the corpus mapping
    
    def createWordIndexMapping(self):
        i=0
        for word in self.wordFreq.keys():
            self.wordIndexMapping[word]=i
            i+=1
        #print(self.wordIndexMapping)
        
    def readAspectSeedWords(self):
        with open(projectSettings+"SeedWords.json") as fd:
            seedWords = json.load(fd)
            for aspect in seedWords["aspects"]:
                self.aspectKeywords[aspect["name"]] = aspect["keywords"]

    def createAspectIndexMapping(self):
        i=0;
        for aspect in self.aspectKeywords.keys():
            self.aspectIndexMapping[aspect]=i
            i+=1
        #print(self.aspectIndexMapping)
                
    def readStopWords(self):
        with open(projectSettings+"stopwords.dat") as fd:
            for stopWord in fd:
                self.stopWords.append(stopWord.strip())
        for stopWord in stopwords.words('english'):
            if stopWord not in self.stopWords:
                self.stopWords.append(stopWord)
        #print(self.stopWords)

    def stemmingStopWRemoval(self, review, vocab):
        ''' Does Following things:
        1. Tokenize review into sentences, and then into words
        2. Remove stopwords, punctuation and stem each word
        3. Add words into vocab 
        4. Make Sentence objects and corresponding Review object
        '''
        reviewObj = Review()
        #copying ratings into reviewObj
        for ratingType, rating in review["Ratings"].items():
            reviewObj.ratings[ratingType] = rating
        reviewObj.reviewId = review["ReviewID"]
        
        stemmer = PorterStemmer()
        reviewContent = review["Content"]
        #TODO: Append title too!
        sentencesInReview = nltk.sent_tokenize(reviewContent)
        puncs = set(string.punctuation) #punctuation marks 
        for sentence in sentencesInReview:
            wordList=[]
            words = nltk.word_tokenize(sentence)
            for word in words:
                if not all(c.isdigit() or c in puncs for c in word):
                    word = word.lower()
                    if word not in self.stopWords:
                        word=stemmer.stem(word.lower())
                        vocab.append(word)
                        wordList.append(word)
            if wordList:
                sentenceObj=Sentence(wordList)
                reviewObj.sentences.append(sentenceObj)
        if reviewObj.sentences:
            self.allReviews.append(reviewObj)
            #print(reviewObj)

    def readReviewsFromJson(self):
        ''' Reads reviews frm the corpus, calls stemmingStopWRemoval
        and creates list of lessFrequentWords (frequency<5)
        '''
        vocab=[]
        for filename in glob.glob(os.path.join(path, '*.json')):
            fd=open(filename)
            data=json.load(fd)
            for review in data["Reviews"]:
                self.stemmingStopWRemoval(review,vocab)
        self.wordFreq = FreqDist(vocab)
        for word,freq in self.wordFreq.items():
            if freq < 5:
                self.lessFrequentWords.add(word)
        for word in self.lessFrequentWords:
            del self.wordFreq[word]
        self.createWordIndexMapping()
        
        #print("Less Frequent Words ",self.lessFrequentWords)
        #print("Vocab ", self.wordFreq.pformat(10000))
                 
    def removeLessFreqWords(self):
        emptyReviews = set()
        for review in self.allReviews:
            emptySentences = set()
            for sentence in review.sentences:
                deleteWords = set()
                for word in sentence.wordFreqDict.keys():
                    if word in self.lessFrequentWords:
                        deleteWords.add(word)
                for word in deleteWords:
                    del sentence.wordFreqDict[word]
                if not sentence.wordFreqDict:
                    emptySentences.add(sentence)
            review.sentences[:] = [x for x in review.sentences if x not in emptySentences]
            if not review.sentences:
                emptyReviews.add(review)  
        self.allReviews[:] = [x for x in self.allReviews if x not in emptyReviews]
