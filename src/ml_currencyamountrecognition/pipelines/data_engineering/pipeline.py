"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .featureEngineering import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=banknote_dataframe,
            inputs=["params:rawData_path", "params:currencies"],
            outputs="banknote_pictures",
            name="banknote_pictures"
        ),
        node(
            func=resize_pictures,
            inputs="banknote_pictures",
            outputs="resized_pictures",
            name="resized_pictures"
        ),
    ])
