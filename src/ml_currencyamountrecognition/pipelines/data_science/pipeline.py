"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes.mlModels import *

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
    ])
