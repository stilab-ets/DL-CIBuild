# pip install hyperopt
#in cmd: git clone https://github.com/hyperopt/hyperopt-sklearn
# in spider: cd hyperopt-sklearn, pip install hpsklearn,pip show hpsklearn
import logging
logging.disable(logging.WARNING)
logging.disable(logging.DEBUG)
from hpsklearn import HyperoptEstimator, svc,random_forest,ada_boost,gaussian_nb
from hyperopt import tpe
from hyperopt.pyll import scope
from hyperopt import hp
import optunity
import optunity.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB 
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score


import ConfigSpace as CS
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import BOHB as BOHB

import Utils
import os
import pandas as pd
from timeit import default_timer as timer
import  GA.GARunner as GARunner
global model,train_set

l = ['newton-cg', 'lbfgs', 'sag','saga']
def _random_state(name, random_state):
    if random_state is None:
        return hp.randint(name, 5)
    else:
        return random_state
    
def decision_tree(name,
                  criterion=None,
                  splitter=None,
                  max_features=None,
                  max_depth=None,
                  min_samples_split=None,
                  min_samples_leaf=None,
                  random_state=None):

    def _name(msg):
        return '%s.%s_%s' % (name, 'sgd', msg)

    rval = scope.sklearn_DecisionTreeClassifier(
        criterion=hp.choice(
            _name('criterion'),
            ['gini', 'entropy']) if criterion is None else criterion,
        splitter=hp.choice(
            _name('splitter'),
            ['best', 'random']) if splitter is None else splitter,
        max_features=hp.choice(
            _name('max_features'),
            ['sqrt', 'log2',
             None]) if max_features is None else max_features,
        max_depth=max_depth,
       
        min_samples_split=scope.int(hp.quniform(
            _name('min_samples_split'),
            2, 10, 1)) if min_samples_split is None else min_samples_split,
       
        min_samples_leaf=scope.int(hp.quniform(
            _name('min_samples_leaf'),
            1, 5, 1)) if min_samples_leaf is None else min_samples_leaf,
        random_state=_random_state(_name('rstate'), random_state),
        )
    return rval

@scope.define
def sklearn_LogisticRegression(*args, **kwargs):
    return LogisticRegression(*args, **kwargs)



def logistic_regression(name):
    def _name(msg):
        return '%s.%s_%s' % (name, 'sgd', msg)

    rval = scope.sklearn_LogisticRegression  (
            penalty=hp.choice(_name('penalty'),['l2','none']),
            #C      =scope.int(hp.quniform(_name('C'),1,5,10)),
            solver =hp.choice(_name('solver'), ['newton-cg', 'lbfgs', 'sag','saga'])
        )
    return rval


class MLWorker(Worker):
    def __init__(self, model, train_set, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.X_train, self.y_train = Utils.preprocess_ML(train_set,"train")

    def compute(self, config, *args, **kwargs):
        #print("params",config)
        classifier = self.model(**config)
        classifier.fit(self.X_train, self.y_train)
        y_pred= classifier.predict(self.X_train)
        entry = Utils.getEntry(self.y_train, y_pred)
        return({
                    'loss': float(1 - entry["AUC"]),  # this is the a mandatory field to run hyperband,   
                    #remember: HpBandSter always minimizes!
                    'info': entry # can be used for any user-defined information - also mandatory
                })
    
    
def construct_ML(ML_params,train_set,model):
    global X_train, y_train
    print("params",ML_params)
    X_train, y_train = Utils.preprocess_ML(train_set,"train")
    classifier = model(**ML_params)
    classifier.fit(X_train, y_train)
    y_pred= classifier.predict(X_train)
    entry = Utils.getEntry(y_train, y_pred)
    print('Best scores:', entry)
    return      {
                'model'   : classifier,#required by GA
                 "entry"  : entry #required by GA
                }
def train_svm(kernel, C, degree, coef0,max_iter):
    """A generic SVM training function, with arguments based on the chosen kernel."""
    if 'linear' in kernel or 'rbf' in kernel:
        model = SVC(kernel=kernel, C=C,max_iter=max_iter)
   # elif 'poly' in kernel: 
      #  model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0,max_iter=max_iter)
   
    return model 

def tune_with_PSO(classifier, n_estimators=None, C=None,kernel=None, degree=None, coef0=None,criterion=1,max_depth=None,algorithm=1,random_state=1,penalty=1,solver=1,max_iter=None):
    
    X_train, y_train = Utils.preprocess_ML(train_set,"train")
    # fit the model
    if 'SVC' in classifier:
        model = train_svm(kernel, C, degree, coef0,int(max_iter))
    elif 'RF' in classifier:
        model = RandomForestClassifier(n_estimators=int(n_estimators),max_depth=int(n_estimators))
    
    if 'DT' in classifier:
        if int(criterion)==1:
            criterion='gini'
        else:
            criterion='entropy'
        model = DecisionTreeClassifier(criterion=criterion,max_depth=max_depth)
     
    if "NB" in classifier:
        model = BernoulliNB()
        
    if "ADA" in classifier:
        if int(algorithm)==1:
            algorithm='SAMME'
        else:
            algorithm='SAMME.R'
        
        if int(random_state)==0:
            random_state=0
        else:
            random_state=None
            
        model = AdaBoostClassifier(random_state=random_state,algorithm=algorithm,n_estimators=int(n_estimators))
    
    if "LR" in classifier:
        if int(penalty)==1:
            penalty='l2'
        else:
            penalty='none'
        
        model=  LogisticRegression(max_iter=int(max_iter),penalty=penalty,solver=l[int(solver)])
        
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    return optunity.metrics.roc_auc(y_train, predictions, positive=True)


def convert2(ML_params):
    for key in ML_params:
        if key=='criterion':
            if int(ML_params[key])==1:
                ML_params[key]='gini'
            else:
                ML_params[key]='entropy'
        
        elif key=='algorithm':
            if int(ML_params[key])==1:
                ML_params[key]='SAMME'
            else:
                ML_params[key]='SAMME.R'
        
        elif key=='random_state':
            if int(ML_params[key])==0:
                ML_params[key]=0
            else:
                ML_params[key]=None
        
        elif key=='solver':
            if(Utils.isInt(ML_params[key])):
                ML_params[key]=l[int(ML_params[key])]
        
        elif key=='penalty':
            if( ML_params[key]==1):
                ML_params[key]='l2'
            else:
                ML_params[key]='none'
               
        elif Utils.isInt(ML_params[key]):
            ML_params[key] = int(ML_params[key])
    
    return ML_params
    
    
def RF_construct(n_estimators=None,criterion=1,max_depth=None):
    return tune_with_PSO(classifier="RF",n_estimators=n_estimators,criterion=criterion,max_depth=max_depth)

def SVC_construct(kernel=None, C=None,  degree=None, coef0=None,max_iter=None):
    return tune_with_PSO(classifier="SVC",kernel=kernel, C=C,  degree=degree, coef0=coef0,max_iter=max_iter)

def DT_construct(criterion=1, max_depth=None):
    return tune_with_PSO(classifier="DT",criterion=criterion, max_depth=max_depth)

def ADA_construct(random_state=1, n_estimators=None,algorithm=1):
    return tune_with_PSO(classifier="ADA",random_state=random_state,algorithm=algorithm,n_estimators=n_estimators)

def NB_construct(random_state=1, n_estimators=None,algorithm=1):
    return tune_with_PSO(classifier="NB")

def LR_construct(max_iter=None, penalty=1,solver=1):
    return tune_with_PSO(classifier="LR",max_iter=max_iter,penalty=penalty,solver=solver)
 


def run_Online (nameAlgo,tunerOption):
    global model,train_set
    rep=2
    config_space = CS.ConfigurationSpace()

    if "SVC" in nameAlgo:
        model=svc
        ML_params = {
                         'classifier':SVC,
                         'params':{
                           'C': [1, 2],
                            'kernel': ['linear','rbf'],
                            'max_iter':     [200, 400, 600, 800, 1000,2000,5000]
                            }
                      }
        params_PSO={
                    'kernel': 
                            {'linear': {'C': [1, 2],'max_iter':[200,5000]},
                                'rbf': {'C': [1, 10],'max_iter':[200,5000]}
                               #'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1],'max_iter':[200,5000]}
                             }
                    }
        fn_pso = SVC_construct
        
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('kernel',  choices=['linear','rbf']))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_iter',  lower=200, upper=5000))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('C',  lower=1, upper=2))
       
        
    elif "DT" in nameAlgo:
        rep=32
        model=decision_tree
        ML_params = {
                        'classifier':DecisionTreeClassifier,
                        'params':{
            			 'criterion':     ['gini', 'entropy'],
                          'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
                        }
                      }
        params_PSO={
                    'criterion':     [1, 2],
                    'max_depth': [10,100]
                    }
        
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('criterion',  choices=['gini', 'entropy']))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth',  lower=10, upper=100))
            
        fn_pso = DT_construct
        
    elif "RF" in nameAlgo:
        rep=32
        model=random_forest
        ML_params = {
                        'classifier':RandomForestClassifier,
                        'params':{
                            'n_estimators':     [50,100,200, 400, 600],
                            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            				'criterion':     ['gini', 'entropy']
                            }
                      }
        params_PSO={
                    'n_estimators': [50,600],
                    'max_depth': [10,100],
                    'criterion':     [1, 2]
                    }

        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_depth',  lower=10, upper=100))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('n_estimators',  lower=50, upper=600))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('criterion',  choices=['gini', 'entropy']))
        fn_pso = RF_construct
        
    elif "ADA" in nameAlgo:
        model=ada_boost
        ML_params = {
                        'classifier':AdaBoostClassifier,
                        'params':{
                            'random_state':     [None,0],
                            'n_estimators':     [50,100,200, 400, 600],
            				'algorithm':     ['SAMME', 'SAMME.R']
                            }
                      }
        params_PSO={
                    'random_state': [0,1],
                    'n_estimators': [50,600],
                    'algorithm':     [1, 2]
                    }
        fn_pso = ADA_construct
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('algorithm',  choices=['SAMME', 'SAMME.R']))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('n_estimators',  lower=50, upper=600))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('random_state',  choices=[None,0]))
        
    elif "NB" in nameAlgo:
        model=gaussian_nb
        ML_params = {
                        'classifier':BernoulliNB,
                        'params':{
                            }
                      }
        params_PSO={}
        fn_pso = NB_construct
   
    elif "LR" in nameAlgo:
        model=logistic_regression
        ML_params = {
                        'classifier':LogisticRegression,
                        'params':{
                           'max_iter':     [200, 400, 600, 800, 1000,2000,5000],
                            'penalty': ['l2','none'],
                            'solver':l
                            }
                      }
        params_PSO={
                    'max_iter': [200,5000],
                    'penalty': [1,2],
                    'solver':     [1, 4]
                    }
        
        fn_pso = LR_construct
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('solver',  choices=l))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('max_iter',  lower=200, upper=5000))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('penalty',  choices=['l2','none']))
       
        ##############################################################################################
    compares = pd.DataFrame(columns =["proj"]+["algo"]+["AUC"]+["time"]+["F1"]+["accuracy"]+["best_params"])
    for file_name in os.listdir("dataset"):
        dataset = pd.read_csv("dataset/"+file_name, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
        dataset.sort_values(by=['gh_build_started_at'], inplace=True)
        X_train, y_train = Utils.preprocess_ML(dataset,"train")
        for i in range(1,rep):
           # try:
            start = timer()
            if "tpe" in tunerOption:
                best_model = HyperoptEstimator( algo=tpe.suggest, 
                                    max_evals=5, 
                                    #trial_timeout=30,
                                    classifier=model(nameAlgo))
            
            if "ga" in tunerOption:
                best_param ,best_model , entry_train = GARunner.generate( 
                                                            ML_params['params'], 
                                                            train_sets[K],
                                                            construct_ML, 
                                                            ML_params['classifier']
                                                        )
           
            if "pso" in tunerOption:
                train_set =  train_sets[K]
                best_params, info, solver_inf = optunity.maximize_structured(fn_pso, params_PSO,  num_evals=50)
                best_params = convert2(best_params)
                #print("the PSO params",best_params)
                res = construct_ML(best_params,train_set,ML_params['classifier'])
                best_model = res["model"]
                
            if "bohb" in tunerOption:
                w = MLWorker(model = ML_params['classifier'], train_set= train_sets[K] , nameserver='127.0.0.1',run_id=nameAlgo)
                w.run(background=True)
                NS = hpns.NameServer(run_id= nameAlgo, host='127.0.0.1', port=None)
                NS.start()
                bohb = BOHB(configspace = config_space, run_id = nameAlgo, nameserver='127.0.0.1')
                res = bohb.run(n_iterations=5)
                bohb.shutdown(shutdown_workers=True)
                NS.shutdown()
                best_params = res.get_id2config_mapping()
                best = res.get_incumbent_id()
                best_model = ML_params['classifier'](**best_params[best]['config'])
                print('Best found configuration:', best_params[best]['config'])
                print('A total of %i unique configurations where sampled.' % len(best_params.keys()))
                print('A total of %i runs where executed.' % len(res.get_all_runs()))
            
            if "RS" in  tunerOption:
                clf = RandomizedSearchCV(ML_params['classifier'](), ML_params['params'], random_state=0)
                search = clf.fit(X_train, y_train)
                best_params = search.best_params_
                res = construct_ML(best_params,train_sets[K],ML_params['classifier'])
                best_model = res["model"]
            ##############################################################################################
            end = timer()
            period = (end - start)
            best_model.fit(X_train, y_train)
            ######################### TRAIN ########################
            y_pred= best_model.predict(X_train)
            entry = {}
            entry["proj"] = file_name
            entry["algo"] = nameAlgo
            entry["AUC"] =  roc_auc_score(y_train,y_pred)
            entry["time"] =  period
            compares = compares.append(entry,ignore_index=True)
            print('train         **************',   )
            ######################### TEST ########################
            y_pred= best_model.predict(X_test)
            entry = {}
            entry["proj"] = file_name
            entry["algo"] = nameAlgo
            entry["AUC"] =  roc_auc_score(y_test,y_pred )
            entry["accuracy"] =  accuracy_score(y_test, y_pred)
            entry["F1"] =  f1_score(y_test, y_pred)
            entry["iter"] = i
            entry["exp"] = (K+1)
            entry["trainSize"] = size_train
            entry["testSize"] = size_test
            entry["trainFailure"] = rate_train
            entry["testFailure"] =rate_test
            results = results.append(entry,ignore_index=True)
            print('test         **************', entry)
            # except Exception as ex:
            #     print("error in "+file_name," ",K," ",nameAlgo)
            #     print(ex)
    results.to_excel(nameAlgo+"_"+tunerOption+"_online.xlsx")
    compares.to_excel("compare_"+nameAlgo +"_"+tunerOption+"_online.xlsx")

run_Online("ADA","RS")

