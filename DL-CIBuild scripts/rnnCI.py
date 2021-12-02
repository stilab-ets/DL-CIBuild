import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def train_classifier(X_train,y_train,network_params):
    # Initialising the RNN
    classifier = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = network_params["nb_units"], return_sequences = True, input_shape = (X_train.shape[1], 1)))
    classifier.add(Dropout(network_params["drop_proba"]))
    # Adding LSTM layer and some Dropout regularisation
    for nbLayesr in range (0,network_params["nb_layers"]):
        classifier.add(LSTM(units = network_params["nb_units"], return_sequences = True))
        classifier.add(Dropout(network_params["drop_proba"]))
    # Adding another LSTM layer without return_sequences
    classifier.add(LSTM(units = network_params["nb_units"]))
    classifier.add(Dropout(network_params["drop_proba"]))
    # Adding the output layer
    classifier.add(Dense(units = 1,activation='sigmoid'))
    # Compiling the RNN
    classifier.compile(optimizer = network_params["optimizer"],
                       loss = 'binary_crossentropy',metrics=["accuracy"])
    # Fitting the RNN to the Training set
    classifier.fit(X_train, y_train, epochs = network_params["nb_epochs"]
                   , batch_size = network_params["nb_batch"],
                   verbose=0)
    return classifier
def predict_with_classifier(classifier,X_test,y_test,proj_name,decision_threshold):
    print(type(decision_threshold))
    decision_threshold=float(decision_threshold)
    predicted_builds = classifier.predict(X_test)
    predicted_builds = (predicted_builds > decision_threshold)
    print('auc=%.2f' % (roc_auc_score(y_test, predicted_builds))
    ,'accuracy =%.2f' % (accuracy_score(y_test, predicted_builds))
    ,'F1=%.2f' % f1_score(y_test,predicted_builds))
    entry = {}
    entry["proj"] = proj_name
    entry["algo"] = "LSTM"
    entry["AUC"] =  roc_auc_score(y_test, predicted_builds)
    entry["accuracy"] =  accuracy_score(y_test, predicted_builds)
    entry["F1"] =  f1_score(y_test,predicted_builds)
    entry["precision"] =  precision_score(y_test,predicted_builds)
    entry["recall"] =  recall_score(y_test,predicted_builds)
    return entry
def train_preprocess(dataset_train,time_step):
    training_set = dataset_train.iloc[:,0:19].values
    X_train = []
    y_train = []
    for i in range(time_step, len(training_set)):
        X_train.append(training_set[i-time_step:i, 0])#0 : we have only one column in training_set
        y_train.append(training_set[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
     #X_train.shape[0] : nbr of lines or observations; X_train.shape[1]:nbr of columns or timestep; 1: nbr of indicators
    return X_train,y_train
def test_preprocess(dataset_train,dataset_test,time_step):
    #Test preprocessing
    real_builds = dataset_test.iloc[:,0:1].values
    dataset_total = pd.concat((dataset_train['build_Failed'], dataset_test['build_Failed']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values
    inputs = inputs.reshape(-1,1)
    X_test = []
    for j in range(time_step, len(inputs)):
        X_test.append(inputs[j-time_step:j, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test,real_builds
def predict(dataset_train,dataset_test,proj_name,network_params):
    X_train,y_train = train_preprocess(dataset_train,network_params["time_step"])
    X_test,real_builds = test_preprocess(dataset_train,dataset_test,network_params["time_step"])
    classifier = train_classifier(X_train,y_train,network_params)
    return predict_with_classifier(classifier,X_test,real_builds,proj_name,network_params["decision_threshold"])