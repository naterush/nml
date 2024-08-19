from collections import Counter
from typing import Literal, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass


# Cat classifier example from coursera
DATA = np.array([
    (
        'P', 'R', 'P', 7.2, 1
    ),
    (
        'O', 'N', 'P', 8.8, 1
    ),
    (
        'O', 'R', 'A', 15, 0
    ),
    (
        'P', 'N', 'P', 9.2, 0
    ),
    (
        'O', 'R', 'P', 8.4, 1
    ),
    (
        'P', 'R', 'A', 7.6, 1
    ),
    (
        'F', 'N', 'A', 11, 0
    ),
    # Wrong
    # [0 1 0 1 0]
    (
        'O', 'R', 'A', 10.2, 1
    ),
    (
        'F', 'R', 'A', 18, 0
    ),
    (
        'F', 'R', 'A', 20, 0
    ),
], dtype='object')  


def one_hot_encode(col):
    unique = np.unique(col)
    new_columns = []
    for u in unique:
        new_columns.append(
            np.where(col == u, 1, 0)
        )
    
    return new_columns


def convert_to_binary(d):
    # Create a copy of the input array
    binary_arr = np.array(d, copy=True)
    
    # Iterate through each column
    all_new_columns = []
    for col in range(binary_arr.shape[1]):
        # Get unique values in the column
        unique_values = np.unique(binary_arr[:, col])

        # If it's already a float column, then do not change it
        if any(isinstance(f, float) for f in unique_values):
            new_columns = [binary_arr[:, col]]
        
        # Ensure there are only two unique values
        elif len(unique_values) == 1:
            new_columns = [np.full((binary_arr.shape[0], 1), 1)]
        elif len(unique_values) == 2:
            new_columns = [np.where(binary_arr[:, col] == unique_values[0], 0, 1)]
        else:
            # Otherwise, we're going to one-hot-encode the algorithm
            new_columns = one_hot_encode(binary_arr[:, col])

        all_new_columns.extend(new_columns)

    new_data = np.array(all_new_columns).T
    return new_data

DATA = convert_to_binary(DATA)

X = DATA[:, :-1]
Y = DATA[:, -1]

# Entropy function
def H(y):
    if len(y) == 0:
        return 0
    p1 = len(y[y == np.unique(y)[0]]) / len(y)
    if p1 == 0 or p1 == 1:
        return 0
    return -p1 * np.log2(p1) - (1 - p1) * np.log2(1 - p1)

def select_next_feature(X, Y):
    rows, cols = X.shape

    root_entropy = H(Y)
    # If we're at the end, we are making a decision node
    if root_entropy == 0:
        return None, None   

    max_decrease, max_decrease_feature, decision_boundry = 0, None, None
    for feature in range(cols):
        unique_values = np.unique(X[:, feature])
        if all(v == 0 or v == 1 for v in unique_values):
            X_sub_1 = X[X[:, feature] == 0]
            Y_sub_1 = Y[X[:, feature] == 0]
            X_sub_2 = X[X[:, feature] == 1]
            Y_sub_2 = Y[X[:, feature] == 1]
            mini_decision_boundry = None
        else:
            # In the case that this is 
            max_decrease_spec, mini_decision_boundry = 0, None
            for val in unique_values:
                X_sub_1 = X[X[:, feature] < val]
                Y_sub_1 = Y[X[:, feature] < val]
                X_sub_2 = X[X[:, feature] >= val]
                Y_sub_2 = Y[X[:, feature] >= val]

                entropy_sub_1 = H(Y_sub_1)
                entropy_sub_2 = H(Y_sub_2)

                weighted_entropy_sub_1 = Y_sub_1.shape[0] / rows * entropy_sub_1
                weighted_entropy_sub_2 = Y_sub_2.shape[0] / rows * entropy_sub_2

                weighted_entropy = weighted_entropy_sub_1 + weighted_entropy_sub_2

                entropy_decrease = root_entropy - weighted_entropy
                if entropy_decrease > max_decrease_spec:
                    max_decrease_spec = entropy_decrease
                    mini_decision_boundry = val

            X_sub_1 = X[X[:, feature] < mini_decision_boundry]
            Y_sub_1 = Y[X[:, feature] < mini_decision_boundry]
            X_sub_2 = X[X[:, feature] >= mini_decision_boundry]
            Y_sub_2 = Y[X[:, feature] >= mini_decision_boundry]

        entropy_sub_1 = H(Y_sub_1)
        entropy_sub_2 = H(Y_sub_2)

        weighted_entropy_sub_1 = Y_sub_1.shape[0] / rows * entropy_sub_1
        weighted_entropy_sub_2 = Y_sub_2.shape[0] / rows * entropy_sub_2

        weighted_entropy = weighted_entropy_sub_1 + weighted_entropy_sub_2

        entropy_decrease = root_entropy - weighted_entropy
        if entropy_decrease > max_decrease:
            max_decrease = entropy_decrease
            max_decrease_feature = feature
            decision_boundry = mini_decision_boundry
    
    return max_decrease_feature, decision_boundry


@dataclass
class DecisionNode:
    label: Literal["0", "1"]
    depth: int = 0

    def __repr__(self):
        return f"Decision: {self.label}"

@dataclass
class TreeNode:
    feature: int
    children: Tuple[Union["TreeNode", DecisionNode], Union["TreeNode", DecisionNode]]
    depth: int
    decision_boundry: Optional[float]

    def __repr__(self):
        return self._repr_helper(0)

    def _repr_helper(self, indent_level):
        indent = '  ' * indent_level
        feature_name = f"Feature {self.feature}"
        decision_boundry = f"(>= {self.decision_boundry})" if self.decision_boundry is not None else ""
        child_repr = [self._repr_child(child, indent_level + 1) for child in self.children]
        return f"{indent}{feature_name} {decision_boundry}\n{indent}├─ 0: {child_repr[0]}\n{indent}└─ 1: {child_repr[1]}"

    def _repr_child(self, child, indent_level):
        if isinstance(child, DecisionNode):
            return child.__repr__()
        else:
            return '\n' + child._repr_helper(indent_level)

def build_tree(X, Y):
    root_feature, decision_boundry = select_next_feature(X, Y)
    if root_feature is None or H(Y) == 0:
        return DecisionNode(label=np.unique(Y)[0])
    
    if decision_boundry is None:
        X_sub_1 = X[X[:, root_feature] == 0]
        Y_sub_1 = Y[X[:, root_feature] == 0]
        X_sub_2 = X[X[:, root_feature] == 1]
        Y_sub_2 = Y[X[:, root_feature] == 1]
    else:
        X_sub_1 = X[X[:, root_feature] < decision_boundry]
        Y_sub_1 = Y[X[:, root_feature] < decision_boundry]
        X_sub_2 = X[X[:, root_feature] >= decision_boundry]
        Y_sub_2 = Y[X[:, root_feature] >= decision_boundry]


    tree_sub_1 = build_tree(X_sub_1, Y_sub_1)
    tree_sub_2 = build_tree(X_sub_2, Y_sub_2)

    return TreeNode(
        feature=root_feature, 
        children=(tree_sub_1, tree_sub_2), 
        depth=max(tree_sub_1.depth, tree_sub_2.depth) + 1,
        decision_boundry=decision_boundry
    )

def predict(tree, x):
    curr_node = tree
    while not isinstance(curr_node, DecisionNode):
        curr_feature_val = x[curr_node.feature]
        if curr_node.decision_boundry is not None:
            curr_feature_val = curr_feature_val >= curr_node.decision_boundry

        curr_node = curr_node.children[curr_feature_val]

    return curr_node.label

def sample_with_replacement(X, Y):
    num_rows = X.shape[0]
    sampled_indices = np.random.randint(0, num_rows, size=num_rows)
    return X[sampled_indices], Y[sampled_indices]

def most_common_item(arr):
    count = Counter(arr)    
    most_common = count.most_common(1)    
    return most_common[0][0] if most_common else None

def predict_forest(trees, x):
    predictions = []
    for tree in trees:
        predictions.append(predict(tree, x))    
    return most_common_item(predictions)

def build_forest(X, Y):

    num_trees = 64
    trees = []
    for _ in range(num_trees):
        X_, Y_ = sample_with_replacement(X, Y)
        tree = build_tree(X, Y)
        print(tree)
        trees.append(tree)

    for index, x in enumerate(X):
        prediction = predict_forest(trees, x)
        print(f"Value {index}, correct prediction {Y[index] == prediction}")



def main():
    build_forest(X, Y)

main()

