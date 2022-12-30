"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes.featureEngineering import *
from .nodes.dataVisualization import *
from .nodes.preprocessing import *


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
            func=split_train_test_data,
            inputs="banknote_pictures",
            outputs=["train_df", "test_df"],
            name="splitting_data"
        ),
        node(
            func=data_augmentation,
            inputs="train_df",
            outputs="augmented_data",
            name="augmented_data"
        ),
        node(
            func=feature_extraction,
            inputs="augmented_data",
            outputs="trainDataFeatures",
            name="dataWithFeaturesTrain"
        ),
        node(
            func=feature_extraction,
            inputs="augmented_data",
            outputs="testDataFeatures",
            name="testDataFeatures"
        ),
        node(
            func=data_normalisation,
            inputs="trainDataFeatures",
            outputs="normalized_trainData",
            name="normalized_data"
        ),
        node(
            func=data_normalisation,
            inputs="testDataFeatures",
            outputs="normalized_testData",
            name="normalized_testData"
        ),
        node(
            func=data_standardisation,
            inputs="normalized_trainData",
            outputs="standardized_trainData",
            name="standardized_trainData"
        ),
        node(
            func=data_standardisation,
            inputs="normalized_testData",
            outputs="standardized_testData",
            name="standardized_testData"
        ),
        node(
            func=label_balanced,
            inputs="standardized_trainData",
            outputs="data_distribution",
            name="data_distribution"
        ),
        node(
            func=currency_balanced,
            inputs="standardized_trainData",
            outputs="currency_distribution",
            name="currency_distribution"
        ),
        node(
            func=outliers_detection,
            inputs="standardized_trainData",
            outputs="images_plot",
            name="outliers_detection"
        ),
        node(
            func=ordinal_data_encoding,
            inputs="standardized_trainData",
            outputs="encoded_trainData",
            name="encoded_trainData"
        ),
        node(
            func=ordinal_data_encoding,
            inputs="standardized_testData",
            outputs="encoded_testData",
            name="encoded_testData"
        ),
        node(
            func=data_preparation_for_MLmodels,
            inputs=["encoded_trainData", "encoded_testData"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="data_for_modelML"
        ),
        node(
            func=data_preparation_for_DeepLearning_models,
            inputs=["encoded_trainData", "encoded_testData"],
            outputs=["X_train_DL", "X_test_DL", "X_val_DL", "y_train_DL", "y_test_DL", "y_val_DL"],
            name="data_for_modelDL"
        ),
    ])
