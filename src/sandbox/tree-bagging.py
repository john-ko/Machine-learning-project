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

    maxDepth = 40
    lowestMaxDepth = LowestMSE()
    nFeatures = 60
    minParent = 8

    for l in range(1,nBags):
        print "bags", l
        Xi,Yi = ml.bootstrapData(Xtr,Ytr, M)

        rforest[l] = dtree.treeRegress()
        rforest[l].train(Xi,Yi,maxDepth=maxDepth)
        YtHat[:,l] = rforest[l].predict(Xtr)[:,0] # predict on training data
        YvHat[:,l] = rforest[l].predict(Xte)[:,0]
        mseT = ((Ytr - YtHat[:,0:l].mean(axis=1))**2).mean()
        mseV = ((Yte - YvHat[:,0:l].mean(axis=1))**2).mean()
        lowestMaxDepth.set(mseV, l, maxDepth, minParent, l)
    

    print "Lowest"
    print lowestMaxDepth
    ###########################

def test_run(xTrainFile, yTrainFile):
    X = np.genfromtxt(xTrainFile,delimiter=",")
    Y = np.genfromtxt(yTrainFile,delimiter=",")
    TEST = np.genfromtxt("/home/john/Downloads/kaggle.X1.test.txt",delimiter=",") 
    M = X.shape[0]

    #maxDepth
    ########################

    nBags = 125

    rforest = [None] * nBags

    maxDepth = 40
    nFeatures = 100
    minParent = 8

    for l in range(nBags):
        print "bags", l
        Xi,Yi = ml.bootstrapData(X,Y, M)

        rforest[l] = dtree.treeRegress()
        rforest[l].train(Xi,Yi,maxDepth=maxDepth)

    mTest = TEST.shape[0]
    predict = np.zeros( (mTest, nBags) )
    for i in range(nBags):
        predict[:,i] = rforest[i].predict(TEST).T[0]
 
    predict = predict[:,0]
    _write_to_kaggle("treebag.csv",predict)

    
def write_to_file(x, y):
    fh = open('MSEprogress.txt','a')    # open file for upload
    fh.write('{},{}\n'.format(x,y)) # output each prediction
    fh.close()

def write_to_kaggle(Y):

    fh = open('kaggle-score.csv', 'w')
    fh.write('ID,Predictions\n')
    for i,k in enumerate(Y):
        fh.write('{},{}\n'.format(i+1,k))
    fh.close()

def _write_to_kaggle(file, Y):

    fh = open(file, 'w')
    fh.write('ID,Predictions\n')
    for i,k in enumerate(Y):
        fh.write('{},{}\n'.format(i+1,k))
    fh.close()

def run_GTB():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
 
    print "running me"
    X = np.genfromtxt("/home/john/Downloads/kaggle.X1.train.txt",delimiter=",") # load the text file
    Y = np.genfromtxt("/home/john/Downloads/kaggle.Y.train.txt",delimiter=",") 
    TEST = np.genfromtxt("/home/john/Downloads/kaggle.X1.test.txt",delimiter=",") 

    #depth = 35 # current lowest
    for depth in [100,135,140,145,150]:
        est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=0, loss='ls').fit(X, Y)
        prediction = est.predict(TEST)
        print 'GTB @ depth ' + str(depth) +" " + str(mean_squared_error(y_test, est.predict(x_test)))
        _write_to_kaggle('test_' + str(depth) + '.csv', prediction)
    #best 0.46050957 max_depth = 25



if __name__ == '__main__':
    test_run("../../../../Downloads/kaggle.X1.train.txt", "../../../../Downloads/kaggle.Y.train.txt")