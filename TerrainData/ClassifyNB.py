import sklearn.naive_bayes as sk

def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
    clf = sk.GaussianNB()
    clf.fit(features_train, labels_train)
    #pred = clf.predict(features_test)

    #accuracy = clf.score(features_test, labels_test)
    #print(accuracy)
    return clf
