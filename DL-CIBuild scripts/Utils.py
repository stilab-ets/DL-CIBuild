
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings # `do not disturbe` mode
warnings.filterwarnings('ignore')
sc = StandardScaler()
from numpy import arange
from numpy import argmax

nbr_rep = 6

nbr_gen = 2
nbr_sol = 2
max_eval = nbr_gen*nbr_sol

with_smote = False 
hybrid_option = False # means smote and threshold moving

if hybrid_option:
    with_smote =True


def getDataset(file_name):
    dataset = pd.read_csv("dataset/"+file_name, 
                          parse_dates=['gh_build_started_at'], 
                          index_col="gh_build_started_at")
    dataset.sort_values(by=['gh_build_started_at'], inplace=True)
    return dataset


# def preprocess_ML_deprecated(dataset,typeofData):
#     X=dataset.iloc[:, 1:19].values
#     y = dataset.iloc[:, 0].values
#     X[:, 3:] = sc.fit_transform(X[:, 3:] )
#     if "train" in typeofData:
#         X, y = SMOTE().fit_resample(X, y)
#     return X,y
    
 # apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')

def getBestThreshold(probs, y_train):
    # keep probabilities for the positive outcome only
    #probs = predicted_builds[:, 1]
    thresholds = arange(0, 1, 0.001)
    # evaluate each threshold
    scores = [roc_auc_score(y_train, to_labels(probs, t)) for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    #print('\nThreshold=%.2f, AUC=%.2f' % (thresholds[ix], scores[ix]))
    return  thresholds[ix]


 

def failureInfo(dataset):
    condition =  dataset['build_Failed'] > 0
    rate = (dataset[condition].shape[0]) /dataset.shape[0]
    size=dataset.shape[0]
    return rate,size

def getEntry(y, predicted_builds):
    entry = {}
    entry["AUC"] =  roc_auc_score(y, predicted_builds)
    entry["accuracy"] =  accuracy_score(y, predicted_builds)
    entry["F1"] =  f1_score(y,predicted_builds)
    return entry

def predict_lstm(classifier,X,y):
    predicted_builds = classifier.predict(X)
    
    if with_smote and not hybrid_option:
        decision_threshold = 0.5
    else:
        decision_threshold = getBestThreshold(predicted_builds, y)
        
    predicted_builds = (predicted_builds >= decision_threshold)
    return getEntry(y, predicted_builds)

def isInt(n):
    try:
        n=int(n)
        return True
    except:
        return False
def online_validation_folds(dataset):
    train_sets=[]
    test_sets =[]
    fold_size = int(len(dataset) * 0.1)
    for i in range(6,11):
        train_sets.append(dataset.iloc[0:(fold_size*(i-1))])
        test_sets.append(dataset.iloc[fold_size*(i-1):(fold_size*i)])
    return  train_sets, test_sets
def frange(start, stop=None, step=None):

    if stop == None:
        stop = start + 0.0
        start = 0.0

    if step == None:
        step = 1.0

    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield ("%g" % start) # return float number
        start = start + step
        
def frange_int(start, stop=None, step=None):

    if stop == None:
        stop = start 
        start = 0

    if step == None:
        step = 1

    while True:
        if step > 0 and start >= stop:
            break
        elif step < 0 and start <= stop:
            break
        yield (start) # return int number
        start = start + step
        

