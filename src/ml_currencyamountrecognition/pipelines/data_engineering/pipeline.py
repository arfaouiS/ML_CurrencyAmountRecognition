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
        node(
            func=data_augmentation,
            inputs="resized_pictures",
            outputs="augmented_data",
            name="augmented_data"
        ),
        node(
            func=feature_extraction,
            inputs="augmented_data",
            outputs="final_data",
            name="final_data"
        ),

    ])
