import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
param_choice_ML = {
        'DT':   {
            #'criterion ':     ['gini', 'entropy'],
            'classifier':DecisionTreeClassifier(),
            'params':{
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
            }
        },
        'RF':   {
             'classifier':RandomForestClassifier(),
             'params':{
                'n_estimators':     [100,200, 400, 600],
                'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
                }
        },
        'LR':  {
             'classifier':LogisticRegression(),
             'params':{
                'max_iter':     [200, 400, 600, 800, 1000,2000],
                'penalty': ['l1', 'l2', 'elasticnet', 'none']
                }
        },
        'SVC':  {
             'classifier':SVC(),
             'params':{
                'C': [1, 10, 100, 1000],
                'kernel': ['linear','rbf'],
                'max_iter':     [200, 400, 600, 800, 1000,2000]
                }
        }
    }
#find the best params for Rails project
dataset = pd.read_csv("dataset/rails.csv", parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
dataset.sort_values(by=['gh_build_started_at'], inplace=True)
X_train = dataset.iloc[:, 1:19].values
y_train = dataset.iloc[:, 0].values
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:] )
best_parameters =[]
for model in param_choice_ML:
    elem = param_choice_ML[model]
    grid_search = GridSearchCV(estimator = elem["classifier"],
                           param_grid =  elem["params"],
                           scoring = 'accuracy',
                           cv = 10)
    grid_search.fit(X_train,y_train)
    best_parameters.append({"model":model,"params":grid_search.best_params_})
    print(model,grid_search.best_score_)