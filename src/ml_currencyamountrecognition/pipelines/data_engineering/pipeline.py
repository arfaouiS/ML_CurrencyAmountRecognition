"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .featureEngineering import *
from .dataVisualization import *


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([

        node(
            func=banknote_dataframe,
            inputs=["params:rawData_path", "params:currency_folder_names"],
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
            outputs="dataWithFeatures",
            name="dataWithFeatures"
        ),
        node(
            func=data_normalisation,
            inputs="dataWithFeatures",
            outputs="normalized_data",
            name="normalized_data"
        ),
        node(
            func=data_standardisation,
            inputs="normalized_data",
            outputs="standardized_data",
            name="standardized_data"
        ),
        node(
            func=images_visualization,
            inputs="standardized_data",
            outputs="images_plot",
            name="images_plot"
        ),
        node(
            func=ordinal_data_encoding,
            inputs="normalized_data",
            outputs="encoded_data",
            name="encoded_data"
        ),

        node(
            func=label_balanced,
            inputs="augmented_data",
            outputs="data_distribution",
            name="data_distribution"
        )
    ])
