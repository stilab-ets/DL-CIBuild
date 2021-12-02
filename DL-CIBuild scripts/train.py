"""
Utility used by the Network class to actually train.
"""
from keras.callbacks import EarlyStopping
import rnnCI

# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)
def train_model(network,file_name,train_set):
    """Train the model"""
    print("*******************",network)
    X_train,y_train = rnnCI.train_preprocess(train_set,network["time_step"])
    classifier      = rnnCI.train_classifier(X_train,y_train,network)
    return rnnCI.predict_with_classifier(classifier,X_train,y_train,file_name,network["decision_threshold"])