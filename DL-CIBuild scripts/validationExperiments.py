import pandas as pd
import os
import LSTM_Tuner
import Utils
import threading
global columns_res,columns_comp
columns_res = ["proj"]+["algo"]+["iter"]+["AUC"]+["accuracy"]+["F1"]+["exp"]

        
def online(tuner):
    if "bohb" in tuner:
        import hpbandster.core.nameserver as hpns
        NS = hpns.NameServer(run_id= "LSTM2", host='127.0.0.10', port=None)
        NS.start()
    results_test = pd.DataFrame(columns =  columns_res)
    results_train = pd.DataFrame(columns =  columns_res)
    for file_name in os.listdir("dataset"):
        dataset = Utils.getDataset(file_name)
        train_sets,test_sets = Utils.online_validation_folds(dataset)
        for k in range (len(train_sets)):
            
            for iteration in range (1,Utils.nbr_rep):
                print(iteration,"*************************************** TRAIN",file_name)
                entry_train  = LSTM_Tuner.evaluate_tuner( tuner, train_sets[k])
                entry_train["iter"] = iteration
                entry_train["proj"] = file_name
                entry_train["exp"] =  k+1
                entry_train["algo"] = "LSTM"
                results_train = results_train.append(entry_train,ignore_index=True)
                print(entry_train)
                best_params = entry_train["params"]
                X,y = LSTM_Tuner.test_preprocess(train_sets[k],test_sets[k],best_params["time_step"])
                #res = LSTM_Tuner.construct_lstm_model(best_params,train_sets[k])
                best_model = entry_train["model"]
                print(iteration,"*************************************** TEST",file_name)
                entry_test = Utils.predict_lstm(best_model,X,y)
                entry_test["iter"] = iteration
                entry_test["best_params"] = best_params
                entry_test["proj"] = file_name
                entry_test["exp"] =  k+1
                entry_test["algo"] = "LSTM"
                print(entry_test)
                results_test = results_test.append(entry_test,ignore_index=True)
    #results_test.to_excel("hybrid"+str(Utils.hybrid_option)+str(Utils.with_smote)+ "_result_test_online_LSTM_params"+tuner+".xlsx")
    results_train.to_excel("hybrid"+str(Utils.hybrid_option)+str(Utils.with_smote)+ "_result_train_online_LSTM_params"+tuner+".xlsx")
    if "bohb" in tuner:
        NS.shutdown()

#online("bohb")

def crossProj(tuner):
    results = pd.DataFrame(columns =  columns_res)
    results_train = pd.DataFrame(columns =  columns_res)
    bellwether="jruby.csv"
    trainset = Utils.getDataset(bellwether)
    for iteration in range (1,Utils.nbr_rep):
        entry_train  = LSTM_Tuner.evaluate_tuner(tuner,trainset)
        best_params = entry_train["params"]
        #best_model = entry_train["model"]
        print(iteration,"*************************************** TRAIN",bellwether)
        entry_train["iter"] = iteration
        entry_train["proj"] = bellwether
        entry_train["algo"] = "LSTM"
        entry_train["params"] = best_params
        results_train = results_train.append(entry_train,ignore_index=True)
        print(entry_train)
        # for file_name in os.listdir("dataset"):
        #     if file_name!=bellwether:
        #         print(file_name)
        #         testset = Utils.getDataset(file_name)
        #         X,y = LSTM_Tuner.test_preprocess(trainset,testset,best_params["time_step"])
        #         entry= Utils.predict_lstm(best_model,X,y)
        #         entry["iter"] = iteration
        #         entry["proj"] = file_name
        #         entry["exp"] =  1
        #         entry["algo"] = "LSTM"
        #         results = results.append(entry,ignore_index=True)
    results.to_excel("corqq_proj_paramf_"+str(Utils.hybrid_option)+str(Utils.with_smote)+"_result_crossProj_"+tuner+"_LSTM.xlsx")
    results_train.to_excel("cross_paramf"+str(Utils.hybrid_option)+str(Utils.with_smote)+"_train_crossProj_"+tuner+"_LSTM.xlsx")
   

#************************************************************ A simple example of how to run GA to get the best params:
#crossProj("ga")

# file_name = "cloudify.csv"
# dataset = Utils.getDataset(file_name)
# train_sets,test_sets =Utils.online_validation_folds(dataset)
# k=0
# entry_train = LSTM_Tuner.evaluate_tuner("tpe", train_sets[k])