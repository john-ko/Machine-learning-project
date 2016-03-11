
def run_tree_regressor():
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.cross_validation import cross_val_score
    from sklearn.cross_validation import train_test_split
    import numpy as np
    from sklearn.ensemble import AdaBoostRegressor
 
    print "running me"
    X = np.genfromtxt("/home/john/Downloads/kaggle.X1.train.txt",delimiter=",") # load the text file
    Y = np.genfromtxt("/home/john/Downloads/kaggle.Y.train.txt",delimiter=",") 
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
     
    rng = np.random.RandomState(1)
 
    for depth in [30,35,40,45]:
        treeAdaBoost =  AdaBoostRegressor(DecisionTreeRegressor(max_depth=depth),n_estimators=120, random_state=rng)
        treeAdaBoost.fit(x_train, y_train)
        y_predicted = treeAdaBoost.predict(x_test) 
        print "adabost @ " + depth + ":", treeAdaBoost.score(x_test, y_test)
    #best 0.46050957 max_depth = 25



if __name__ == '__main__':
    run_tree_regressor()