import pandas as pd
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
import os

models = {
           "DT" : DecisionTreeClassifier(max_depth=10),
           "RF" : RandomForestClassifier(n_estimators=100, max_depth=50),
           "NB" : GaussianNB(),
           "LR" : LogisticRegression(max_iter=200,penalty='l2'),
           "SVC": SVC(C=10,kernel='linear',max_iter=2000)
        }
def online_validation_sets(dataset):
    train_sets=[]
    test_sets =[]
    fold_size = int(len(dataset) * 0.1)
    for i in range(6,11):
        train_sets.append(dataset.iloc[0:(fold_size*(i-1))])
        test_sets.append(dataset.iloc[fold_size*(i-1):(fold_size*i)])
    return train_sets, test_sets
def lunchML(train_dataset,test_dataset,proj_name,exp):
    results=[]
    X_train = train_dataset.iloc[:, 1:19].values
    y_train = train_dataset.iloc[:, 0].values
    X_test = test_dataset.iloc[:, 1:19].values
    y_test = test_dataset.iloc[:, 0].values
    # Feature Scaling
    sc = StandardScaler()
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:] )
    X_test[:, 3:]  = sc.transform(X_test[:, 3:] )
    X_train, y_train = SMOTE().fit_resample(X_train, y_train)
    for model_name in models:
        if "DT" in model_name or  "RF" in model_name :
            iteration=32
        else:
            iteration=2
        for i in range(1,iteration):
            models[model_name].fit(X_train,y_train)
            y_test_pred = models[model_name].predict(X_test)
            print(model_name,'   AUC=%.2f' % (roc_auc_score(y_test, y_test_pred)),
                  '   F1=%.2f' % f1_score(y_test,y_test_pred),
              ' accuracy=%.2f' % accuracy_score(y_test,y_test_pred)
              , ' precision=%.2f' % precision_score(y_test,y_test_pred)
              , ' recall=%.2f' % recall_score(y_test,y_test_pred)
             )
            entry = {}
            entry["proj"] = proj_name
            entry["algo"] = model_name
            entry["AUC"] =  roc_auc_score(y_test, y_test_pred)
            entry["accuracy"] =  accuracy_score(y_test, y_test_pred)
            entry["F1"] =  f1_score(y_test,y_test_pred)
            entry["precision"] =  precision_score(y_test,y_test_pred)
            entry["recall"] =  recall_score(y_test,y_test_pred)
            results.append(entry)
    return results
#****************************************************************** online_validation**
results2 = pd.DataFrame(columns = ["proj"]+["algo"]+["AUC"]+["accuracy"]+["F1"]+["precision"] +["recall"])
for file_name in os.listdir("dataset"):
    print('**************',file_name)
    dataset = pd.read_csv("dataset/"+file_name, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
    dataset.sort_values(by=['gh_build_started_at'], inplace=True)
    train_sets,test_sets = online_validation_sets(dataset)
    for k in range (len(train_sets)):
        results2 = results2.append(lunchML(train_sets[k],test_sets[k],file_name,k+1))
results2.to_excel("online_ML.xlsx")
#*************************** CROSS proj
results = pd.DataFrame(columns = ["proj"]+["algo"]+["AUC"]+["accuracy"]+["F1"]+["precision"]+["recall"])
bellwether="jruby.csv"
trainset = pd.read_csv("dataset/"+bellwether, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
trainset.sort_values(by=['gh_build_started_at'], inplace=True)
for file_name in os.listdir("dataset"):
    if bellwether != file_name:
        print(file_name)
        testset = pd.read_csv("dataset/"+file_name, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
        testset.sort_values(by=['gh_build_started_at'], inplace=True)
        train_dataset= trainset.iloc[0:len(trainset)]
        test_dataset =testset.iloc[0:len(testset)]
        results = results.append(lunchML(train_dataset,test_dataset,file_name,1))
results.to_excel("crossProj_ML.xlsx")
