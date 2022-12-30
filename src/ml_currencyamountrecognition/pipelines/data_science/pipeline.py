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
