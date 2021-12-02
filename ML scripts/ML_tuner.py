import logging
logging.disable(logging.WARNING)
logging.disable(logging.DEBUG)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import Utils
from timeit import default_timer as timer
import  GA.GARunner as GARunner
import pandas as pd
import os
from sklearn.metrics import log_loss
################################################
global X_train_global, y_train_global,tpe_last_params,tpe_model_init,tpe_last_model,tpe_last_entry
solvers = ['newton-cg', 'lbfgs', 'sag','saga']
global sc
#################################################
def preprocess_ML(dataset,typeofData):
    global sc
    X=dataset.iloc[:, 1:19].values
    if "train" in typeofData:
         sc = StandardScaler().fit(X[:, 3:])

    y = dataset.iloc[:, 0].values
    X[:, 3:] = sc.transform(X[:, 3:]) 
    if "train" in typeofData:
       X, y = SMOTE().fit_resample(X, y)
    return X,y

def construct_ML(ML_params,X_train, y_train,model):
    #print("params in construct_ML ",ML_params)
    if model == SVC:
        ML_params["probability"]=True
    
    
    classifier = model(**ML_params)
    classifier = classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_train)
    entry = Utils.getEntry(y_train, y_pred)
    yhat = classifier.predict_proba(X_train)
    
    loss = log_loss(y_train,yhat)
    entry['validation_loss']=loss
    print("loss",loss)
    return      {
                 'validation_loss': loss
                ,'model'   : classifier,#required by GA
                 "entry"  : entry #required by GA
                }

def GA_train_fn (ML_params, data):
    return construct_ML(ML_params,data['X_train'],data['y_train'],data['model'])

def tune_ML (nameAlgo, X_train, y_train):
    global X_train_global, y_train_global,tpe_model_init
    X_train_global = X_train
    y_train_global = y_train

    if "SVC" in nameAlgo:
        C_list = [1, 2]
        kern = ['linear','rbf']
        max_ters = [200, 400, 600, 800, 1000,2000,5000]
        ML_params = {
                         'classifier':SVC,
                         'params':{
                           'C': C_list,
                            'kernel': kern,
                            'max_iter':     max_ters
                            }
                      }
       
        
    elif "DT" in nameAlgo:
        
        criter =  ['gini', 'entropy']
        max_dep = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
        split = [2, 10]
        leaf = [1,2,3,4,5]
        feat = ['sqrt', 'log2',  None]
        ML_params = {
                        'classifier':DecisionTreeClassifier,
                        'params':{
            			 'criterion':   criter  ,
                          'max_depth': max_dep,
                          'min_samples_split': split,
                          'min_samples_leaf':leaf, 
                          'max_features':feat
                        }
                      }
       
      
        
    elif "RF" in nameAlgo:
        crit = ['gini', 'entropy']
        n_est =  [50,100,200, 400, 600,1000,5000]
        max_dep = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500,1000, None]
        split = [2, 10,50,100]
        leaf = [1,2,3,4,5,50]
        feat = ['sqrt', 'log2',  None]
        ML_params = {
                        'classifier':RandomForestClassifier,
                        'params':{
                            'n_estimators':   n_est ,
                            'max_depth': max_dep,
            				'criterion':    crit ,
                            'min_samples_split': split,
                            'min_samples_leaf':leaf, 
                            'max_features':feat
                            
                            }
                      }
       
    elif "ADA" in nameAlgo:
        rand = [None,0]
        est = [50,100,200, 400, 600]
        alg = ['SAMME', 'SAMME.R']
        learning_rate = [0.06,0.7,0.8,0.9,1.0]
        ML_params = {
                        'classifier':AdaBoostClassifier,
                        'params':{
                            'random_state': rand    ,
                            'n_estimators':    est ,
            				'algorithm':   alg ,
                            'learning_rate':learning_rate
                            }
                      }
      
        
    elif "NB" in nameAlgo:
        ML_params = {
                        'classifier':BernoulliNB,
                        'params':{
                            }
                      }
    
   
    elif "LR" in nameAlgo:
        max_i = [200, 400, 600, 800, 1000,2000,5000]
        penal = ['l2','none']
        ML_params = {
                        'classifier':LogisticRegression,
                        'params':{
                           'max_iter':   max_i  ,
                            'penalty': penal,
                            'solver':solvers
                            }
     
    ##############################################################################################
    start = timer()
    params_fn = {
            'X_train':X_train,
            'y_train':y_train,
            'model':ML_params['classifier']
            }
        best_params ,best_model , best_entry = GARunner.generate(  ML_params['params'], GA_train_fn,params_fn  )
    ##############################################################################################
    end = timer()
    period = (end - start)
    best_entry["time"] = period
    best_entry["best_params"] = best_params
    best_entry["model"] = best_model
    return best_entry

columns_res = ["proj"]+["algo"]+["exp"]+["iter"]+["AUC"]+["accuracy"]+["F1"]
def online(algo,tuner,option_TM="TM"):
    results = pd.DataFrame(columns =  columns_res)
    for file_name in os.listdir("dataset"):
        print("********************************************",file_name,"********************************************")
        dataset = Utils.getDataset(file_name)
        train_sets,test_sets = Utils.online_validation_folds(dataset)
        for K in range (len(train_sets)):
            X_train, y_train = preprocess_ML(train_sets[K],"train")
            X_test, y_test =   preprocess_ML(test_sets[K],"test")
            for iteration in range (1,Utils.nbr_rep):
                print("******",iteration,"*********")
                res = tune_ML(algo,tuner, X_train,y_train)
                best_model = res["model"]
                ######################### TEST ########################
                if(option_TM=="TM"):
                    ####################### Threshold moving ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    predicted_proba = best_model.predict_proba(X_test)
                    best_threshold = Utils.getBestThreshold(predicted_proba[:,1], y_test)
                    y_pred = (predicted_proba [:,1] >= best_threshold).astype('int')
                else:
                    y_pred= best_model.predict(X_test)
                
                entry  = Utils.getEntry(y_test, y_pred)
                entry["proj"] = file_name
                entry["algo"] = algo
                entry["iter"] = iteration
                entry["exp"] = (K+1)
                results = results.append(entry,ignore_index=True)
                print("Score in test",entry)
    results.to_excel("test_online_ML_"+algo+"_"+tuner+"_Threshold_"+option_TM+".xlsx")

def cross(algo,tuner,option_TM="TM"):
    results = pd.DataFrame(columns =  columns_res)
    bellwether="jruby.csv"
    trainset = Utils.getDataset(bellwether)
    X_train, y_train = preprocess_ML(trainset,"train")
    if "bohb" in tuner:
        import hpbandster.core.nameserver as hpns
        NS = hpns.NameServer(run_id= algo, host='127.0.0.1', port=None)
        NS.start()
    for file_name in os.listdir("dataset"):
        if file_name!=bellwether:
            print("********************************************",file_name,"********************************************")
            testset = Utils.getDataset(file_name) 
            X_test, y_test =   preprocess_ML(testset,"test")
            for iteration in range (1,Utils.nbr_rep):
                res = tune_ML(algo,tuner, X_train,y_train)
                best_model = res["model"] 
                ######################### TEST ########################
                if(option_TM=="TM"):
                                ####################### Threshold moving ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                                predicted_proba = best_model.predict_proba(X_test)
                                best_threshold = Utils.getBestThreshold(predicted_proba[:,1], y_test)
                                y_pred = (predicted_proba [:,1] >= best_threshold).astype('int')
                else:
                    y_pred= best_model.predict(X_test)
                                
                entry  = Utils.getEntry(y_test, y_pred)
                entry["proj"] = file_name
                entry["algo"] = algo
                entry["iter"] = iteration
                results = results.append(entry,ignore_index=True)
                print("Score",entry)
    results.to_excel("cross_ML_"+algo+"_"+tuner+"TM_"+option_TM+".xlsx")
    if "bohb" in tuner:
        NS.shutdown()

    
