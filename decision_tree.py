from typing import Literal, Tuple, Union
import numpy as np
from dataclasses import dataclass

@dataclass
class DecisionNode:
    label: Literal[0, 1]

    def __repr__(self):
        return f"Decision: {self.label}"

@dataclass
class DecisionTree:
    feature: int
    decision_boundary: float
    children: Tuple[Union["DecisionTree", "DecisionNode"], Union["DecisionTree", "DecisionNode"]]

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

def calculate_entropy(y):
    if len(y) == 0:
        return 0
    p1 = np.sum(y) / len(y)
    if p1 == 0:
        return 0
    return - p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

def calculate_gini(y):
    if len(y) == 0:
        return 0
    p1 = np.sum(y) / len(y)
    return 1 - p1**2 - (1 - p1)**2

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

    
def build_decision_tree(
        X, 
        y,
        height=0
    ):

    n, m = X.shape
    base_entropy = calculate_entropy(y)

    max_entropy_decrease, best_split_feature, best_decision_boundary = 0, -1, None
    for feature in range(m):
        for decision_boundary in get_decision_boundaries(X, feature):
            X_left, y_left, X_right, y_right = split(X, y, feature, decision_boundary)

            new_entropy = len(X_left) / len(X) * calculate_entropy(y_left) + len(X_right) / len(X) * calculate_entropy(y_right)
            entropy_decrease = base_entropy - new_entropy

            if entropy_decrease > max_entropy_decrease:
                max_entropy_decrease = entropy_decrease
                best_split_feature, best_decision_boundary = feature, decision_boundary

    if best_split_feature == -1 or best_decision_boundary is None or height > 5:
        # Pick the item with the best split
        result = 1 if np.sum(y) > .5 * len(y) else 0
        return DecisionNode(result)
    
    X_left, y_left, X_right, y_right = split(X, y, best_split_feature, best_decision_boundary)
    left_node = build_decision_tree(X_left, y_left, height=height+1)
    right_node = build_decision_tree(X_right, y_right, height=height+1)

    return DecisionTree(best_split_feature, best_decision_boundary, (left_node, right_node))

def predict(X, decision_tree: Union[DecisionNode, DecisionTree]):

    predictions = []
    for x in X:
        curr_node = decision_tree
        while not isinstance(curr_node, DecisionNode):
            is_left = x[curr_node.feature] <= curr_node.decision_boundary
            curr_node = curr_node.children[0] if is_left else curr_node.children[1]

        predictions.append(curr_node.label)
    
    return np.array(predictions)