


from copy import deepcopy
import math
from typing import List, Union


class NDArray:

    def __init__(self, data=None):
        if data is None:
            data = []
        
        # Assure it's a list, or a list of lists
        self.is_vector = all(isinstance(x, (int, float)) for x in data)
        self.is_matrix = not self.is_vector

        if self.is_matrix:
            lengths = set([len(x) for x in data])
            if len(lengths) > 1:
                raise ValueError('Matrixs must be rectangular')
        
        if self.is_vector:
            self.shape = (len(data), 0)
        else:
            self.shape = (len(data), len(data[0]) if len(data) > 0 else 0)

        # Save the data. 
        self.data = data

    @property
    def T(self):
        if self.is_vector:
            return NDArray(self.data)
        else:
            if len(self.data) == 0:
                return NDArray()
            
            columns = [[x] for x in self.data[0]]
            for row in self.data[1:]:
                for index, value in enumerate(row):
                    columns[index].append(value)

            return NDArray(columns)
        
    def __repr__(self):
        return f"{self.data}"
    
    def __apply(self, func):
        if self.is_vector:
            return NDArray([func(x) for x in self.data])
        else:
            return NDArray([[func(x) for x in row] for row in self.data])

    def __truediv__(self, b):
        return self.__apply(lambda x: x / b)



def array(data):
    return NDArray(data)

def dot(one: Union[NDArray, List], two: Union[NDArray, List]):
    if isinstance(one, list):
        one = NDArray(one)
    if isinstance(two, list):
        two = NDArray(two)

    if one.shape != two.shape:
        raise ValueError(f'To dot vectors, they must have same shape. Got {one.shape} and {two.shape}')
    
    if one.is_vector:
        return sum(map(lambda x: x[0] * x[1], zip(one.data, two.data)))
    else:
        return sum(
            map(
                lambda x: dot(x[0], x[1]) ,
                zip(one.data, two.data)
            )
        )

def matmul(one: Union[NDArray, List], two: Union[NDArray, List]):
    if isinstance(one, list):
        one = NDArray(one)
    if isinstance(two, list):
        two = NDArray(two)

    shape1 = one.shape
    shape2 = two.shape

    if shape1[1] != shape2[0]:
        raise ValueError(f'To matmul matrixes, they must have matching inner shapes. Got {one.shape} and {two.shape}')

    # Rows in the first one (easy to get) with the columns in the second one
    # which is just rows in two.T
    new_data = [
        [
            0 for _ in range(shape2[1])
        ]
        for _ in range(shape1[0])
    ]
    for row_index, row in enumerate(one.data):
        for col_index, col in enumerate(two.T.data):
            new_data[row_index][col_index] = dot(row, col)

    return NDArray(new_data)


def is_triangular(one: NDArray):
    lower, upper = True, True
    for index, row in enumerate(one.data):
        lower = lower and all(x == 0 for x in row[:index])
    for index, row in enumerate(one.data):
        upper = upper and all(x == 0 for x in row[len(row) - index:])

    return lower or upper

    


def det(one: Union[NDArray, List]):
    if isinstance(one, list):
        one = NDArray(one)

    shape = one.shape
    if shape[0] != shape[1]:
        raise ValueError(f"To calculate det, must be square, no {shape}")
    
    # We use the following properties:
    # 1. Switch two rows ⇒ negates the determinant.
    # 2. Multiply row by scalar ⇒ multiplies the determinant by that value. This feels very reasonable.
    # 3. Adding a multiple of one row to another does not change determinant. With a little work, this makes sense, as it’s just shearing the resulting paralellogram! Is this by a constant?????
    # 4. The determinant of a triangular matrix is the product of it’s diagonals.

    curr = one

    while not is_triangular(curr):
        found = False
        for i, row in enumerate(curr.data):
            # Find the first row that has 
            # Then, go to the row above it, and find what multiple we need to make to delete 
            # this fella
            non_lower_triangular = any(x != 0 for x in row[:i])
            if non_lower_triangular:
                found = True
                break

        if not found:
            break


        # row has zeros before it should have zeros. So we're going to 
        # add a combination of other rows above to it to get rid of the
        # zeros before the diagonal...
        curr_row = deepcopy(row)
        while any(x != 0 for x in curr_row[:i]):
            for j, element in enumerate(curr_row[:i]):
                if element != 0:
                    adjustment_row = curr.data[j]
                    scaling_factor = element / adjustment_row[j]
                    to_subtract = [x * scaling_factor for x in adjustment_row]
                    curr_row = [x - y for x, y in zip(curr_row, to_subtract)]
        
        curr.data[i] = curr_row

    det = 1
    for i, row in enumerate(one.data):
        det *= row[i]

    return det

def inv():
    pass


def normalize(m: NDArray):
    return m / det(m)


            
m = array([[1, 1, 2, 1], [4, 2, 5, 3], [5, 2, 2, 5], [4, 4, 4, 2]])
m1 = array([[1, 2, 2], [1, 2, 2]])
v = array([1, 2, 3])
print(det(normalize(m)))




