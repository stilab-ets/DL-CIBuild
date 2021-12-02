#from hyperopt.pyll.base import scope 
#from hyperopt.pyll.stochastic import sample 
from hyperopt import hp,Trials,STATUS_OK ,fmin,tpe,rand
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout
from keras.callbacks import EarlyStopping
import optunity
import optunity.metrics
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import Utils
import  GA.GARunner as GARunner
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB as BOHB
from timeit import default_timer as timer


def train_preprocess(dataset_train,time_step):
    training_set = dataset_train.iloc[:,0:19].values
    if Utils.with_smote:
        X= training_set
        y= dataset_train.iloc[:,0].values
        X, y = SMOTE().fit_resample(X, y)
        training_set = X
    
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
    y_test = dataset_test.iloc[:,0:1].values
    dataset_total = pd.concat((dataset_train['build_Failed'], dataset_test['build_Failed']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values
    inputs = inputs.reshape(-1,1)
    X_test = []
    for j in range(time_step, len(inputs)):
        X_test.append(inputs[j-time_step:j, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_test,y_test

def get_threshold_list(dataset):
    cdt =  dataset['build_Failed'] > 0
    failure_rate = (dataset[cdt].shape[0] /dataset.shape[0])
    return list(Utils.frange(0.01,max(1,failure_rate), 0.1))

class LSTMWorker(Worker):
    def __init__(self,  train_set, **kwargs):
        super().__init__(**kwargs)
        self.train_set= train_set

    def compute(self, config, *args, **kwargs):
        res = construct_lstm_model(config,self.train_set)
        return({
                    'loss': float(res["validation_loss"]),  # this is the a mandatory field to run hyperband,   
                    #remember: HpBandSter always minimizes!
                    'info': res["entry"] # can be used for any user-defined information - also mandatory
                })

def construct_lstm_model (network_params,train_set):
    X_train,y_train = train_preprocess(train_set,network_params["time_step"])# need to preprocess each time to tune the time_step
    drop = round(network_params["drop_proba"])
    # Initialising the RNN
    classifier = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    classifier.add(LSTM(units = network_params["nb_units"], return_sequences = True, input_shape = (X_train.shape[1], 1)))
    classifier.add(Dropout(drop))
    # Adding LSTM layer and some Dropout regularisation
    for nbLayesr in range (0,network_params["nb_layers"]):
        classifier.add(LSTM(units = network_params["nb_units"], return_sequences = True))
        classifier.add(Dropout(drop))
    # Adding another LSTM layer without return_sequences
    classifier.add(LSTM(units = network_params["nb_units"]))
    classifier.add(Dropout(drop))
    # Adding the output layer
    classifier.add(Dense(units = 1,activation='sigmoid'))
    # Compiling the RNN
    classifier.compile(optimizer = network_params["optimizer"],
                       loss = 'binary_crossentropy',metrics=["accuracy"])
    
    es = EarlyStopping(monitor='loss',mode='min', verbose=1,patience=10)
    
     # Fitting the RNN to the Training set
    result =  classifier.fit(X_train, y_train, epochs = network_params["nb_epochs"]
                   , batch_size = network_params["nb_batch"],
                   verbose=0, callbacks=[es])
    
    # Get the lowest validation loss of the training epochs
    validation_loss = np.amin(result.history['loss']) 
    # Get prediction probs
    entry = Utils.predict_lstm(classifier,X_train,y_train)
    entry['validation_loss']=validation_loss
    return      {
                'validation_loss'  : validation_loss, #required by TPE,GA
                'model'   : classifier#required by GA
                ,"entry"  : entry #required by GA
                }
global data
global global_params
global global_model
global global_entry

def train_lstm_with_hyperopt(network_params):
    global global_params,global_model,global_entry
    res = construct_lstm_model(network_params,data)
    global_params = network_params
    global_model = res["model"]
    global_entry = res["entry"]
    return {
            'loss'  : res['validation_loss'], #required by hyperopt
            'status': STATUS_OK,  #required by hyperopt
            }
    
def convert_from_PSO(network_params):
    for key in network_params:
        if key=='optimizer':
            if int(network_params[key])==1:
                 network_params[key] ='adam'
            else:
                network_params[key] ='rmsprop'
        elif not 'drop_proba' in key and not 'decision_threshold'   in key:
            network_params[key] = int(network_params[key])
    
    return network_params

def fn_lstm_pso(drop_proba=0.01,nb_units=32,nb_epochs=2,nb_batch=4,nb_layers=1,optimizer=1,time_step=30):
    
    if int(optimizer)==1:
        optimizer = 'adam'
    else:
        optimizer = 'rmsprop'
    
    network_params = {
             'nb_units':  int(nb_units),
            'nb_layers':  int(nb_layers),
            'optimizer':  optimizer,
            'time_step':  int(time_step),
            'nb_epochs':  int(nb_epochs),
            'nb_batch':   int(nb_batch),
            'drop_proba': drop_proba
            #'decision_threshold'       :  decision_threshold
        }
    res = construct_lstm_model(network_params,data)
    return 1-float(res["validation_loss"])
    
def evaluate_tuner(tuner_option, train_set):
    global data
    data = train_set
    #########################################
    nb_units =  list(Utils.frange_int(32,64, 32))#[64]#,128,256
    nb_epochs = [4,5,6]#list(Utils.frange_int(5,10, 1))#list(Utils.frange_int(5,25, 5))#15,20,25,,10
    nb_batch =[4,8,16,32, 64]#,, . power of 2
    nb_layers = [1,2,3,4]
    optimizers = [ 'adam','rmsprop']#,
    time_steps = list(Utils.frange_int(30,61, 1))
    drops = list(Utils.frange_int(0.01,0.21, 0.01))
    #threshold_list = get_threshold_list(data)
    space_tpe = {
         'drop_proba'               : hp.choice('drop_proba',drops),
         'nb_units'                 : hp.choice('nb_units', nb_units),
         'nb_epochs'                : hp.choice('nb_epochs',nb_epochs),
         'nb_batch'                 : hp.choice('nb_batch',nb_batch),
         'nb_layers'                : hp.choice('nb_layers',nb_layers),
         'optimizer'                : hp.choice('optimizer',optimizers),
         'time_step'                : hp.choice('time_step',time_steps)
       #  'decision_threshold'       : hp.choice('decision_threshold', threshold_list)
        }
    ##########################################################
    start = timer()

    if "tpe" in tuner_option:
        trials = Trials()
        fmin(train_lstm_with_hyperopt,  space_tpe,    algo=tpe.suggest, max_evals= Utils.max_eval,  trials= trials)
        best_params = global_params
        best_model  = global_model
        entry_train = global_entry
    
    elif "ga" in tuner_option:
        
        rnn_param_choices = {
            'nb_units':   nb_units,
            'nb_layers':  nb_layers,
            'optimizer':  optimizers,
            'time_step':  time_steps,
            'nb_epochs':  nb_epochs,
            'nb_batch':   nb_batch,
            'drop_proba': drops
           # 'decision_threshold'       :  threshold_list
        }
        best_params ,best_model , entry_train = GARunner.generate(rnn_param_choices, construct_lstm_model, data)

    elif "pso" in tuner_option:
        params_PSO={
             'nb_units':   [nb_units[0],nb_units[len(nb_units)-1]],
            'nb_layers':  [nb_layers[0],nb_layers[len(nb_layers)-1]],
            'optimizer':  [1,2],
            'time_step':  [time_steps[0],time_steps[len(time_steps)-1]],
            'nb_epochs':  [nb_epochs[0],nb_epochs[len(nb_epochs)-1]],
            'nb_batch':   [nb_batch[0],nb_batch[len(nb_batch)-1]],
            'drop_proba': [drops[0],drops[len(drops)-1]]
            #'decision_threshold'       :  [float(threshold_list[0]),float(threshold_list[len(threshold_list)-1])],
                    }
        best_params, info, solver_inf = optunity.maximize_structured(fn_lstm_pso, params_PSO,  num_evals= Utils.max_eval)
        best_params = convert_from_PSO(best_params)
        res = construct_lstm_model(best_params,data)
        entry_train = res["entry"]
        
        best_model = res["model"]
    
    elif "bohb" in tuner_option:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_units',  lower=nb_units[0], upper=nb_units[len(nb_units)-1]))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_layers',  lower=nb_layers[0], upper=nb_layers[len(nb_layers)-1]))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('optimizer',  choices=optimizers))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('time_step',  lower=time_steps[0], upper=time_steps[len(time_steps)-1]))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_epochs',  lower=nb_epochs[0], upper=nb_epochs[len(nb_epochs)-1]))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_batch',  lower=nb_batch[0], upper=nb_batch[len(nb_batch)-1]))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('drop_proba', 
                                                                      lower=drops[0],
                                                                      upper=drops[len(drops)-1]))
        # config_space.add_hyperparameter(CS.UniformFloatHyperparameter('decision_threshold', 
        #                                                               lower=float(threshold_list[0]),
        #                                                               upper=float(threshold_list[len(threshold_list)-1])))
        
        id_ = "LSTM"
        w = LSTMWorker(train_set= data , nameserver='127.0.0.1',run_id= id_)
        w.run(background=True)
        bohb = BOHB(configspace = config_space, run_id = id_, nameserver='127.0.0.1',min_budget = 1,max_budget = Utils.nbr_sol)
        res = bohb.run(n_iterations=  Utils.nbr_gen)
        bohb.shutdown(shutdown_workers=True)
        best_params = res.get_id2config_mapping()
        best = res.get_incumbent_id()
        print('A total of %i unique configurations where sampled.' % len(best_params.keys()))
        print('A total of %i runs where executed.' % len(res.get_all_runs()))
        best_params = best_params[best]['config']
        res = construct_lstm_model(best_params,data)
        entry_train = res["entry"]
        best_model = res["model"]
        
    elif "rs" in tuner_option:
        trials = Trials()
        fmin(train_lstm_with_hyperopt,  space_tpe,    algo=rand.suggest, max_evals=  Utils.max_eval,  trials= trials)
        best_params = global_params
        entry_train = global_entry
        best_model  = global_model
        
    elif "default" in tuner_option:
         best_params = {
             'nb_units':  64,
            'nb_layers':  3,
            'optimizer':  'adam',
            'time_step':  30,
            'nb_epochs':  10,
            'nb_batch':   64,
            'drop_proba': 0.1
            #'decision_threshold'    :  0.5
          }
         res = construct_lstm_model(best_params, data)
         entry_train = res["entry"]
         best_model  = res["model"]
    ##########################################################################################
    end = timer()
    period = (end - start)
    entry_train["time"] = period
    entry_train["params"] = best_params
    entry_train["model"]  = best_model
    return entry_train
  
# import hpbandster.core.nameserver as hpns
# NS = hpns.NameServer(run_id= "LSTM", host='127.0.0.1', port=None)
# NS.start()
# file_name = "cloudify.csv"
# dataset = Utils.getDataset(file_name)
# train_sets,test_sets =Utils.online_validation_folds(dataset)
# k=0
# entry_train_ga = evaluate_tuner("ga", train_sets[k])
# X,y = test_preprocess(train_sets[k],test_sets[k],entry_train_ga["params"]["time_step"])
# entry_test = Utils.predict_lstm(entry_train_ga["model"],X,y)
# NS.shutdown()

# bellwether="jruby.csv"
# trainset = Utils.getDataset(bellwether)
# entry_train_cross  = evaluate_tuner("ga",trainset)
# best_params = entry_train_cross["params"]

# X,y =  test_preprocess(trainset,dataset,best_params["time_step"])
# best_model = entry_train_cross["model"]
# entry_test_scross= Utils.predict_lstm(best_model,X,y)
    
