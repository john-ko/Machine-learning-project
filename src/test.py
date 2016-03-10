import numpy as np
import sys
sys.path.append("../")
import mltools as ml
import mltools.linear
import matplotlib.pyplot as plt
import mltools.dtree as dtree

class LowestMSE:

    def __init__(self):
        self.nBags = 0
        self.maxDepth = 0
        self.minParent = 0
        self.mse = 100
    
    def set(self, mse, nBags, maxDepth, minParent, test=None):
        if mse < self.mse:
            if test:
                    write_to_file(test, mse)
            self.mse = mse
            self.nBags = nBags
            self.maxDepth = maxDepth
            self.minParent = minParent
            print mse, nBags, maxDepth, minParent

    def __str__(self):
        return "Lowest MSE: " + str(self.mse) + ", nbags:" + str(self.nBags) + ", maxdepth:" + str(self.maxDepth) + ", minparent:" + str(self.minParent)




def setup_code(xTrainFile, yTrainFile):
    X1 = np.genfromtxt(xTrainFile,delimiter=",")
    Y = np.genfromtxt(yTrainFile,delimiter=",")
    Xtr,Xte,Ytr,Yte = ml.splitData(X1,Y,0.80)
    
    M = Xtr.shape[0]
    Mv= Xte.shape[0]

    #maxDepth
    ########################

    nBags = 6000
    YtHat = np.zeros((M,nBags))
    YvHat = np.zeros((Mv,nBags))
    rforest = [None] * nBags

    maxDepth = 30
    lowestMaxDepth = LowestMSE()
    nFeatures = 60
    minParent = 8

    for l in range(1,nBags):
        print "bags", l
        Xi,Yi = ml.bootstrapData(Xtr,Ytr, M)

        rforest[l] = dtree.treeRegress()
        rforest[l].train(Xi,Yi,maxDepth=30)
        YtHat[:,l] = rforest[l].predict(Xtr)[:,0] # predict on training data
        YvHat[:,l] = rforest[l].predict(Xte)[:,0]
        mseT = ((Ytr - YtHat[:,0:l].mean(axis=1))**2).mean()
        mseV = ((Yte - YvHat[:,0:l].mean(axis=1))**2).mean()
        lowestMaxDepth.set(mseV, l, 30, minParent, l)
    

    print "Lowest"
    print lowestMaxDepth
    ###########################
    
def write_to_file(x, y):
    fh = open('MSEprogress.txt','a')    # open file for upload
    fh.write('{},{}\n'.format(x,y)) # output each prediction
    fh.close()

setup_code("../../../../Downloads/kaggle.X1.train.txt", "../../../../Downloads/kaggle.Y.train.txt")