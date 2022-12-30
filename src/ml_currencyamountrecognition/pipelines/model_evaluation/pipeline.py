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
    ])
