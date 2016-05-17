
def run_GTB():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import mean_squared_error
    import numpy as np
 
    print "running me"
    X = np.genfromtxt("/home/john/Downloads/kaggle.X1.train.txt",delimiter=",") # load the text file
    Y = np.genfromtxt("/home/john/Downloads/kaggle.Y.train.txt",delimiter=",") 
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
     
    rng = np.random.RandomState(1)
 
    #depth = 35 # current lowest
    for depth in [30,35,40,45,50]:
        est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=depth, random_state=0, loss='ls').fit(x_train, y_train)
        print 'GTB @ depth ' + str(depth) + str(mean_squared_error(y_test, est.predict(x_test)))
    #best 0.46050957 max_depth = 25



if __name__ == '__main__':
    run_GTB()