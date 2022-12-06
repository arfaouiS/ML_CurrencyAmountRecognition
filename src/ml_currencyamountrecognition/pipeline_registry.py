"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from ml_currencyamountrecognition.pipelines import data_engineering as de


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    return pipelines

    data_engineering_pipeline = de.create_pipeline()

    return {
        "__default__": data_engineering_pipeline,
        "de": data_engineering_pipeline,
    }
