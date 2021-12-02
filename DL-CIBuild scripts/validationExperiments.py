import pandas as pd
import os
import rnnCI
import runGA

def getDataset(file_name):
    dataset = pd.read_csv("dataset/"+file_name, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
    dataset.sort_values(by=['gh_build_started_at'], inplace=True)
    return dataset
def online_validation_sets(dataset):
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
def get_threshold_list(dataset):
    cdt =  dataset['build_Failed'] > 0
    failure_rate = (dataset[cdt].shape[0] /dataset.shape[0])
    return list(frange(0.01,max(1,(failure_rate+0.2)), 0.1))
    
def online(network_params):
    #P.S we defined for each fold validation the specific params provided by GA
    results = pd.DataFrame(columns =  ["proj"]+["algo"]+["AUC"]+["accuracy"]+["precision"]+["recall"]+["F1"])
    for file_name in os.listdir("dataset"):
        dataset = getDataset(file_name)
        train_sets,test_sets = online_validation_sets(dataset)   
        for k in range (len(train_sets)):
            rnn_param_choices['decision_threshold']=get_threshold_list(train_sets[k])
            print(rnn_param_choices['decision_threshold'])
            best_params = runGA.generate(generations, population, rnn_param_choices,file_name,train_sets[k])
            for exp in range (1,32):
                entry= rnnCI.predict(train_sets[k],test_sets[k],file_name,best_params)
                results = results.append(entry,ignore_index=True)
        print("*******************",file_name)
    results.to_excel("result_online_LSTM.xlsx")
def crossProj():
    #P.S we defined for each project validation the specific params provided by GA
    results = pd.DataFrame(columns =  ["proj"]+["algo"]+["AUC"]+["accuracy"]+["precision"]+["recall"]+["F1"])
    bellwether="jruby.csv"
    trainset = getDataset(bellwether)
    rnn_param_choices['decision_threshold']=get_threshold_list(trainset)
    best_params = runGA.generate(generations, population, rnn_param_choices,bellwether,trainset)
    #Train the model
    X_train,y_train = rnnCI.train_preprocess(trainset,best_params["time_step"])
    for exp in range (1,32):
        classifier = rnnCI.train_classifier(X_train,y_train,best_params)
        for file_name in os.listdir("dataset"):
            if file_name!=bellwether:
                testset = getDataset(file_name)
                X_test,real_builds = rnnCI.test_preprocess(trainset,testset,best_params["time_step"])
                entry= rnnCI.predict_with_classifier(classifier,X_test,real_builds,file_name,best_params["decision_threshold"])
                results = results.append(entry,ignore_index=True)
                print(file_name)
    results.to_excel("result_crossProj_LSTM.xlsx")
#************************************************************ A simple example of how to run GA to get the best params:
#Get the data =  first fold of cloudify project
file_name = 'cloudify.csv'
fold = 0
dataset = getDataset(file_name)
#fix threshold possible values depending on the failure rate
train_sets,test_sets = online_validation_sets(dataset)
th_list = get_threshold_list(train_sets[fold])
#*****************************************************************Params for GA
#possible params values for GA
rnn_param_choices = {
    'nb_units':   [64,128,256],
    'nb_layers':  [2,3,4],
    'optimizer':  ['rmsprop', 'adam'],
    'time_step':  [30,60,90,120],
    'nb_epochs':  [5,10,20,25],
    'nb_batch':   [16,32,64],
    'drop_proba': [0.1,0.2,0.3],
}
rnn_param_choices['decision_threshold']=th_list
generations = 10  # Number of times to evole the population.
population =  20  # Number of candidates in each generation.
#The other params like mutation_proba are set in Optimizer.py
#RUN GA
best_params = runGA.generate(generations, population, rnn_param_choices,file_name,train_sets[fold])
#execute the expriments based on the best params for example for the fold=1
results = pd.DataFrame(columns =  ["proj"]+["algo"]+["AUC"]+["accuracy"]+["precision"]+["recall"]+["F1"])
entry= rnnCI.predict(train_sets[fold],test_sets[fold],file_name,best_params.network)
results = results.append(entry,ignore_index=True)
results.to_excel("example_online_"+file_name+"fold"+str(fold)+"LSTM.xlsx")
