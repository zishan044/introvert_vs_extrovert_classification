from kedro.pipeline import Node, Pipeline
from .nodes import preprocess_data


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([Node(func=preprocess_data, inputs='raw', outputs='feature_data', name='preproces_data_node')])
