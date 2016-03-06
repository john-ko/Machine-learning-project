


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
            print nBags, maxDepth, minParent

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

    nBags = 25
    YtHat = np.zeros((M,nBags))
    YvHat = np.zeros((Mv,nBags))
    rforest = [None] * nBags

    maxDepth = 100
    lowestMaxDepth = LowestMSE()
    nFeatures = 60
    minParent = 8

    for l in range(10, nBags):
        for mDepth in range(10, maxDepth):

            Xi,Yi = ml.bootstrapData(Xtr,Ytr, M)

            rforest[l] = dtree.treeRegress()
            rforest[l].train(Xi,Yi,maxDepth=mDepth,nFeatures=nFeatures, minParent=minParent)
            YtHat[:,l] = rforest[l].predict(Xtr)[:,0] # predict on training data
            YvHat[:,l] = rforest[l].predict(Xte)[:,0]
            mseT = ((Ytr - YtHat[:,0:l].mean(axis=1))**2).mean()
            mseV = ((Yte - YvHat[:,0:l].mean(axis=1))**2).mean()
            lowestMaxDepth.set(mseV, l, mDepth, minParent, mDepth)
    

    print "Lowest"
    print lowestMaxDepth
    ###########################
    
def write_to_file(x, y):
    fh = open('MSEprogress.txt','a')    # open file for upload
    fh.write('ID,Prediction\n')         # output header line
    fh.write('{},{}\n'.format(x,y)) # output each prediction
    fh.close()

setup_code("/home/john/Downloads/kaggle.X1.train.txt", "/home/john/Downloads/kaggle.y.train.txt")
