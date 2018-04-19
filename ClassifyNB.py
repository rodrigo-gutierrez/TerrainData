#import sklearn.naive_bayes as sk
import sklearn.svm as sk
from time import time

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    #clf = sk.GaussianNB()
    clf = sk.SVC(kernel='rbf', C=100000.0, gamma=10.0)

    print "Training start"
    timer = time()
    clf.fit(features_train, labels_train)
    print "Training time: ", round(time() - timer, 3), "s"

    #print "Predicting start"
    #timer = time()
    #clf.predict(features_test)
    #print "Predicting time: ", round(time() - timer, 3), "s"

    #accuracy = clf.score(features_test, labels_test)
    #print(accuracy)
    return clf
