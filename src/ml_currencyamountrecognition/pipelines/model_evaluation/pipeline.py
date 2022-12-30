"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .modelEvaluation import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=mlModel_evaluation,
            inputs=["svm_model", "X_test", "y_test"],
            outputs="svmScores",
            name="svm_evaluation"
        ),
        node(
            func=mlModel_evaluation,
            inputs=["nb_model", "X_test", "y_test"],
            outputs="nbScores",
            name="naiveBayes_evaluation"
        ),
        node(
            func=mlModel_evaluation,
            inputs=["knn_model", "X_test", "y_test"],
            outputs="knnScores",
            name="knn_evaluation"
        ),
        node(
            func=mlModel_evaluation,
            inputs=["dtc_model", "X_test", "y_test"],
            outputs="dtcScores",
            name="dtc_evaluation"
        ),
        node(
            func=dlModel_evaluation,
            inputs=["X_test_DL", "y_test_DL", "model_LSTM1"],
            outputs="LSTM1_scores"
        ),
        node(
            func=dlModel_evaluation,
            inputs=["X_test_DL", "y_test_DL", "model_LSTM2"],
            outputs="LSTM2_scores"
        ),
        node(
            func=dlModel_evaluation,
            inputs=["X_test_DL", "y_test_DL", "model_CNN1"],
            outputs="CNN1_scores"
        ),
        node(
            func=dlModel_evaluation,
            inputs=["X_test_DL", "y_test_DL", "model_CNN2"],
            outputs="CNN2_scores"
        ),
        node(
            func=dlModel_evaluation,
            inputs=["X_test_DL", "y_test_DL", "model_feedward_neural_network"],
            outputs="feedwardNeuralNetwork_scores"
        ),
    ])
