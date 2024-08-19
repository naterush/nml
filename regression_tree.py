# Regression tree: decision tree for prediction continiously valued variables
# See here: https://www.stat.cmu.edu/~cshalizi/350-2006/lecture-10.pdf

from typing import Literal, Tuple, Union
import numpy as np
from dataclasses import dataclass

from nateml.loss import mse

@dataclass
class DecisionNode:
    label: float

    def __repr__(self):
        return f"Decision: {self.label}"

@dataclass
class SplitNode:
    feature: int
    decision_boundary: float
    children: Tuple[Union["SplitNode", "DecisionNode"], Union["SplitNode", "DecisionNode"]]

    def __repr__(self):
        return self._repr_helper(0)

    def _repr_helper(self, indent_level):
        indent = '  ' * indent_level
        feature_name = f"Feature {self.feature}"
        child_repr = [self._repr_child(child, indent_level + 1) for child in self.children or []]
        return f"{indent}{feature_name} \n{indent}├─ 0: {child_repr[0]}\n{indent}└─ 1: {child_repr[1]}"

    def _repr_child(self, child, indent_level):
        if isinstance(child, DecisionNode):
            return child.__repr__()
        else:
            return '\n' + child._repr_helper(indent_level)
        
RegressionTree = Union[SplitNode, DecisionNode]


def split(X, y, feature, decision_boundary):
    col = X[:, feature]

    left = col <= decision_boundary
    right = col > decision_boundary

    X_left, y_left = X[left], y[left]
    X_right, y_right = X[right], y[right]

    return X_left, y_left, X_right, y_right

def get_decision_boundaries(X, feature):
    col = X[:, feature]
    uniques = np.unique(col)

    if len(uniques) < 2:
        return []
    
    return [(x + y) / 2 for x, y in zip(uniques, uniques[1:])]

    
def build_regression_tree(
        X, 
        y,
        height=0,
        max_height=5
    ):

    n, m = X.shape

    min_avg_variance, best_split_feature, best_decision_boundary = None, None, None
    for feature in range(m):
        for decision_boundary in get_decision_boundaries(X, feature):
            X_left, y_left, X_right, y_right = split(X, y, feature, decision_boundary)

            new_avg_variance = len(X_left) / len(X) * np.var(y_left) + len(X_right) / len(X) * np.var(y_right)

            if min_avg_variance is None or min_avg_variance > new_avg_variance:
                min_avg_variance, best_split_feature, best_decision_boundary = new_avg_variance, feature, decision_boundary

    if best_split_feature is None or best_decision_boundary is None or height >= max_height:
        result = np.average(y)
        return DecisionNode(result)
    
    X_left, y_left, X_right, y_right = split(X, y, best_split_feature, best_decision_boundary)
    left_node = build_regression_tree(X_left, y_left, height=height+1, max_height=max_height)
    right_node = build_regression_tree(X_right, y_right, height=height+1, max_height=max_height)

    return SplitNode(best_split_feature, best_decision_boundary, (left_node, right_node))

def predict(X, decision_tree: Union[DecisionNode, SplitNode]):

    predictions = []
    for x in X:
        curr_node = decision_tree
        while not isinstance(curr_node, DecisionNode):
            is_left = x[curr_node.feature] <= curr_node.decision_boundary
            curr_node = curr_node.children[0] if is_left else curr_node.children[1]

        predictions.append(curr_node.label)
    
    return np.array(predictions)

def loss(model, X, y):
    predictions = predict(X, model)
    return mse(predictions, y)