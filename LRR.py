
# coding: utf-8

# In[7]:

import numpy as np
from scipy import optimize
import json
import random
import logging
modelDataDir = "modelData/"

class LRR:
    def __init__(self):
        self.vocab=[]
        self.vocab = self.loadDataFromFile("vocab.json")
        
        self.aspectKeywords={}
        self.aspectKeywords = self.loadDataFromFile("aspectKeywords.json")
        
        #word to its index in the corpus mapping
        self.wordIndexMapping={} 
        self.createWordIndexMapping()
        
        #aspect to its index in the corpus mapping
        self.aspectIndexMapping={} 
        self.reverseAspIndexmapping={}
        self.createAspectIndexMapping()
        
        #list of Wd matrices of all reviews
        self.wList=[]
        self.wList = self.loadDataFromFile("wList.json")

        #List of ratings dictionaries belonging to review class
        self.ratingsList=[]
        self.ratingsList = self.loadDataFromFile("ratingsList.json")
        
        #List of Review IDs
        self.reviewIdList=[]
        self.reviewIdList = self.loadDataFromFile("reviewIdList.json")
        
        #number of reviews in the corpus
        self.R = len(self.reviewIdList)
        
        #breaking dataset into 3:1 ratio, 3 parts for training and 1 for testing
        self.trainIndex = random.sample(range(0, self.R), int(0.75*self.R))
        self.testIndex = list(set(range(0, self.R)) - set(self.trainIndex))

        #number of aspects
        self.k = len(self.aspectIndexMapping)
        
        #number of training reviews in the corpus
        self.Rn = len(self.trainIndex)
        
        #vocab size
        self.n = len(self.wordIndexMapping)
        
        #delta - is simply a number
        self.delta = 1.0
        
        #matrix of aspect rating vectors (Sd) of all reviews - k*Rn
        self.S = np.empty(shape=(self.k, self.Rn), dtype=np.float64)
        
        #matrix of alphas (Alpha-d) of all reviews - k*Rn
        #each column represents Aplha-d vector for a review
        self.alpha = np.random.dirichlet(np.ones(self.k), size=1).reshape(self.k, 1)
        for i in range(self.Rn-1):
            self.alpha = np.hstack((self.alpha, np.random.dirichlet(np.ones(self.k), size=1).reshape(self.k, 1)))
        
        #vector mu - k*1 vector
        self.mu = np.random.dirichlet(np.ones(self.k), size=1).reshape(self.k, 1)
        
        #matrix Beta for the whole corpus (for all aspects, for all words) - k*n matrix
        self.beta = np.random.uniform(low=-0.1, high=0.1, size=(self.k, self.n))
        
        #matrix sigma for the whole corpus - k*k matrix
        #Sigma needs to be positive definite, with diagonal elems positive
        '''self.sigma = np.random.uniform(low=-1.0, high=1.0, size=(self.k, self.k))
        self.sigma = np.dot(self.sigma, self.sigma.transpose())
        print(self.sigma.shape, self.sigma)
        '''
        
        #Following is help taken from: 
        #https://stats.stackexchange.com/questions/124538/
        W = np.random.randn(self.k, self.k-1)
        S = np.add(np.dot(W, W.transpose()), np.diag(np.random.rand(self.k)))
        D = np.diag(np.reciprocal(np.sqrt(np.diagonal(S))))
        self.sigma = np.dot(D, np.dot(S, D))
        self.sigmaInv=np.linalg.inv(self.sigma)
        
        ''' testing for positive semi definite
        if(np.all(np.linalg.eigvals(self.sigma) > 0)): #whether is positive semi definite
            print("yes")
        print(self.sigma)
        '''
        # setting up logger
        self.logger = logging.getLogger("LRR")
        self.logger.setLevel(logging.INFO)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler("lrr.log")
        formatter = logging.Formatter('%(asctime)s %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
    def createWordIndexMapping(self):
        i=0
        for word in self.vocab:
            self.wordIndexMapping[word]=i
            i+=1
        #print(self.wordIndexMapping)
    
    def createAspectIndexMapping(self):
        i=0;
        for aspect in self.aspectKeywords.keys():
            self.aspectIndexMapping[aspect]=i
            self.reverseAspIndexmapping[i]=aspect
            i+=1
        #print(self.aspectIndexMapping)
        
    def loadDataFromFile(self,fileName):
        with open(modelDataDir+fileName,'r') as fp:
            obj=json.load(fp)
            fp.close()
            return obj
    
    #given a dictionary as in every index of self.wList, 
    #creates a W matrix as was in the paper
    def createWMatrix(self, w):
        W = np.zeros(shape=(self.k, self.n))
        for aspect, Dict in w.items():
            for word, freq in Dict.items():
                W[self.aspectIndexMapping[aspect]][self.wordIndexMapping[word]]=freq
        return W
    
    #Computing aspectRating array for each review given Wd->W matrix for review 'd'
    def calcAspectRatings(self,Wd):
        Sd = np.einsum('ij,ij->i',self.beta,Wd).reshape((self.k,))
        try:                                                                                                                     
            Sd = np.exp(Sd)
        except Exception as inst:
            self.logger.info("Exception in calcAspectRatings : %s", Sd)
        return Sd
    
    def calcMu(self): #calculates mu for (t+1)th iteration
        self.mu = np.sum(self.alpha, axis=1).reshape((self.k, 1))/self.Rn
        
    def calcSigma(self, updateDiagonalsOnly): #update diagonal entries only
        self.sigma.fill(0)
        for i in range(self.Rn):
            columnVec = self.alpha[:, i].reshape((self.k, 1))
            columnVec = columnVec - self.mu
            if updateDiagonalsOnly:
                for k in range(self.k):
                    self.sigma[k][k] += columnVec[k]*columnVec[k]
            else:
                self.sigma = self.sigma + np.dot(columnVec, columnVec.transpose())
        for i in range(self.k):
                self.sigma[i][i] = (1.0+self.sigma[i][i])/(1.0+self.Rn)
        self.sigmaInv=np.linalg.inv(self.sigma)
    
    def calcOverallRating(self,alphaD,Sd):
        return np.dot(alphaD.transpose(),Sd)[0][0]
        
    def calcDeltaSquare(self):
        self.delta=0.0
        for i in range(self.Rn):
            alphaD=self.alpha[:,i].reshape((self.k, 1))
            Sd=self.S[:,i].reshape((self.k, 1))
            Rd=float(self.ratingsList[self.trainIndex[i]]["Overall"])
            temp=Rd-self.calcOverallRating(alphaD,Sd)
            try:
                self.delta+=(temp*temp)
            except Exception:
                self.logger.info("Exception in Delta calc")
        self.delta/=self.Rn
    
    def maximumLikelihoodBeta(self,x,*args):
        beta = x
        beta=beta.reshape((self.k,self.n))
        innerBracket = np.empty(shape=self.Rn)
        for d in range(self.Rn):
            tmp = 0.0
            rIdx = self.trainIndex[d] #review index in wList
            for i in range(self.k):
                W = self.createWMatrix(self.wList[rIdx])
                tmp += self.alpha[i][d]*np.dot(beta[i, :].reshape((1, self.n)), W[i, :].reshape((self.n, 1)))[0][0]
            innerBracket[d] = tmp - float(self.ratingsList[rIdx]["Overall"])
        mlBeta=0.0
        for d in range(self.Rn):
            mlBeta+=innerBracket[d] * innerBracket[d]
        return mlBeta/(2*self.delta)
    
    def gradBeta(self,x,*args):
        beta=x
        beta=beta.reshape((self.k,self.n))
        gradBetaMat=np.empty(shape=((self.k,self.n)),dtype='float64')
        innerBracket = np.empty(shape=self.Rn)
        for d in range(self.Rn):
            tmp = 0.0
            rIdx = self.trainIndex[d] #review index in wList
            for i in range(self.k):
                W = self.createWMatrix(self.wList[rIdx])
                tmp += self.alpha[i][d]*np.dot(beta[i, :].reshape((1, self.n)), W[i, :].reshape((self.n, 1)))[0][0]
            innerBracket[d] = tmp - float(self.ratingsList[rIdx]["Overall"])
        
        for i in range(self.k):
            beta_i=np.zeros(shape=(1,self.n))
            for d in range(self.Rn):
                rIdx = self.trainIndex[d] #review index in wList
                W = self.createWMatrix(self.wList[rIdx])
                beta_i += innerBracket[d] * self.alpha[i][d] *  W[i, :]
            gradBetaMat[i,:]=beta_i
        return gradBetaMat.reshape((self.k*self.n, ))
            
    def calcBeta(self):
        beta, retVal, flags=optimize.fmin_l_bfgs_b(func=self.maximumLikelihoodBeta,x0=self.beta,fprime=self.gradBeta,args=(),m=5,maxiter=15000)
        converged = True
        if flags['warnflag']!=0:
            converged = False
        self.logger.info("Beta converged : %d", flags['warnflag'])
        return beta.reshape((self.k,self.n)), converged
                
    def maximumLikelihoodAlpha(self, x, *args):
        alphad=x
        alphad=alphad.reshape((self.k, 1))
        rd,Sd,deltasq,mu,sigmaInv=args
        temp1=(rd-np.dot(alphad.transpose(),Sd)[0][0])
        temp1*=temp1
        temp1/=(deltasq*2)
        temp2=(alphad-mu)
        temp2=np.dot(np.dot(temp2.transpose(),sigmaInv),temp2)[0][0]
        temp2/=2
        return temp1+temp2
    
    def gradAlpha(self, x,*args):
        alphad=x
        alphad=alphad.reshape((self.k, 1))
        rd,Sd,deltasq,mu,sigmaInv=args
        temp1=(np.dot(alphad.transpose(),Sd)[0][0]-rd)*Sd
        temp1/=deltasq
        temp2=np.dot(sigmaInv,(alphad-mu))
        return (temp1+temp2).reshape((self.k,))
    
    def calcAlphaD(self,i):
        alphaD=self.alpha[:,i].reshape((self.k,1))
        rIdx = self.trainIndex[i]
        rd=float(self.ratingsList[rIdx]["Overall"])
        Sd=self.S[:,i].reshape((self.k,1))
        Args=(rd,Sd,self.delta,self.mu,self.sigmaInv)
        bounds=[(0,1)]*self.k
        #self.gradf(alphaD, *Args)
        alphaD, retVal, flags=optimize.fmin_l_bfgs_b(func=self.maximumLikelihoodAlpha,x0=alphaD,fprime=self.gradAlpha,args=Args,bounds=bounds,m=5,maxiter=15000)
        converged = True
        if flags['warnflag']!=0:
            converged = False
        self.logger.info("Alpha Converged : %d", flags['warnflag'])
        #Normalizing alphaD so that it follows dirichlet distribution
        alphaD=np.exp(alphaD)
        alphaD=alphaD/(np.sum(alphaD))
        return alphaD.reshape((self.k,)), converged
    
    '''
    def getBetaLikelihood(self):
        likelihood=0
        return self.lambda*np.sum(np.einsum('ij,ij->i',self.beta,self.beta))
    '''
                
    def dataLikelihood(self):
        likelihood=0.0
        for d in range(self.Rn):
            rIdx = self.trainIndex[d]
            Rd=float(self.ratingsList[rIdx]["Overall"])
            W=self.createWMatrix(self.wList[rIdx])
            Sd=self.calcAspectRatings(W).reshape((self.k, 1))
            alphaD=self.alpha[:,d].reshape((self.k, 1))
            temp=Rd-self.calcOverallRating(alphaD,Sd)
            try:                                                                                                                 
                likelihood+=(temp*temp)
            except Exception:
                self.logger.debug("Exception in dataLikelihood")
        likelihood/=self.delta
        return likelihood
    
    def alphaLikelihood(self):
        likelihood=0.0
        for d in range(self.Rn):
            alphad=self.alpha[:,d].reshape((self.k, 1))
            temp2=(alphad-self.mu)
            temp2=np.dot(np.dot(temp2.transpose(),self.sigmaInv),temp2)[0]
            likelihood+=temp2
        try:
            likelihood+=np.log(np.linalg.det(self.sigma))
        except FloatingPointError:
            self.logger.debug("Exception in alphaLikelihood: %f", np.linalg.det(self.sigma))
        return likelihood
    
    def calcLikelihood(self):
        likelihood=0.0
        likelihood+=np.log(self.delta) #delta likelihood
        likelihood+=self.dataLikelihood() #data likelihood - will capture beta likelihood too
        likelihood+=self.alphaLikelihood() #alpha likelihood
        return likelihood
    
    def EStep(self):
        for i in range(self.Rn):
            rIdx = self.trainIndex[i]
            W=self.createWMatrix(self.wList[rIdx])
            self.S[:,i]=self.calcAspectRatings(W)
            alphaD, converged = self.calcAlphaD(i)
            if converged:
                self.alpha[:,i]=alphaD
            self.logger.info("Alpha calculated")
            
    def MStep(self):
        likelihood=0.0
        self.calcMu()
        self.logger.info("Mu calculated")
        self.calcSigma(False)
        self.logger.info("Sigma calculated : %s " % np.linalg.det(self.sigma))
        likelihood+=self.alphaLikelihood() #alpha likelihood
        self.logger.info("alphaLikelihood calculated") 
        beta,converged=self.calcBeta()
        if converged:
            self.beta=beta
        self.logger.info("Beta calculated")
        likelihood+=self.dataLikelihood() #data likelihood - will capture beta likelihood too
        self.logger.info("dataLikelihood calculated")
        self.calcDeltaSquare()
        self.logger.info("Deltasq calculated")
        likelihood+=np.log(self.delta) #delta likelihood
        return likelihood
            
    def EMAlgo(self, maxIter, coverge):
        self.logger.info("Training started")
        iteration = 0
        old_likelihood = self.calcLikelihood()
        self.logger.info("initial calcLikelihood calculated, det(Sig): %s" % np.linalg.det(self.sigma)) 
        diff = 10.0
        while(iteration<min(8, maxIter) or (iteration<maxIter and diff>coverge)):
            self.EStep()
            self.logger.info("EStep completed") 
            likelihood = self.MStep()
            self.logger.info("MStep completed")
            diff = (old_likelihood-likelihood)/old_likelihood
            old_likelihood=likelihood
            iteration+=1
        self.logger.info("Training completed")
    
    def testing(self):
        for i in range(self.R-self.Rn):
            rIdx = self.testIndex[i]
            W = self.createWMatrix(self.wList[rIdx])
            Sd = self.calcAspectRatings(W).reshape((self.k,1))
            overallRating = self.calcOverallRating(self.mu,Sd)
            print("ReviewId-",self.reviewIdList[rIdx])
            print("Actual OverallRating:",self.ratingsList[rIdx]["Overall"])
            print("Predicted OverallRating:",overallRating)
            print("Actual vs Predicted Aspect Ratings:")
            for aspect, rating in self.ratingsList[rIdx].items():
                if aspect != "Overall" and aspect.lower() in self.aspectIndexMapping.keys():
                    r = self.aspectIndexMapping[aspect.lower()]
                    print("Aspect:",aspect," Rating:",rating, "Predic:", Sd[r])
            if overallRating > 3.0:
                print("Positive Review")
            else:
                print("Negative Review")
            
np.seterr(all='raise')
lrrObj = LRR()
lrrObj.EMAlgo(maxIter=10, coverge=0.0001)
lrrObj.testing()


# In[ ]:



