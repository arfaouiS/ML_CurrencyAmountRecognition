"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes.mlModels import *
from .nodes.dlModels import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=svmModel,
            inputs=["X_train", "y_train", "params:svmGridparams"],
            outputs="svm_model",
            name="svm_model"
        ),
        node(
            func=naiveBayesModel,
            inputs=["X_train", "y_train", "params:nbGridparams"],
            outputs="naiveBayesModel",
            name="naiveBayesModel"
        ),
        node(
            func=kNeighborsClassifier,
            inputs=["X_train", "y_train", "params:knnGridparams"],
            outputs="kNeighborsClassifier",
            name="kNeighborsClassifier"
        ),
        node(
            func=decisionTreeClassifier,
            inputs=["X_train", "y_train", "params:decisionTreeGridparams"],
            outputs="decisionTreeClassifier",
            name="decisionTreeClassifier"
        ),
        node(
            func=construct_LSTM_model1,
            inputs=["X_train_DL", "X_val_DL", "y_train_DL", "y_val_DL"],
            outputs="model_LSTM1",
            name="model_LSTM1"
        ),
        node(
            func=construct_LSTM_model2,
            inputs=["X_train_DL", "X_val_DL", "y_train_DL", "y_val_DL"],
            outputs="model_LSTM2",
            name="model_LSTM2"
        ),
        node(
            func=construct_CNN_model1,
            inputs=["X_train_DL", "X_val_DL", "y_train_DL", "y_val_DL"],
            outputs="model_CNN1",
            name="model_CNN1"
        ),
        node(
            func=construct_CNN_model2,
            inputs=["X_train_DL", "X_val_DL", "y_train_DL", "y_val_DL"],
            outputs="model_CNN2",
            name="model_CNN2"
        ),
        node(
            func=construct_feedward_neural_network_model,
            inputs=["X_train_DL", "X_val_DL", "y_train_DL", "y_val_DL"],
            outputs="model_feedward_neural_network",
            name="model_feedward_neural_network"
        ),
    ])
