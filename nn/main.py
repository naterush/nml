from typing import List
import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 

def sigmoid_prime(act):
    return act * (1 - act)

def relu(x):
    return np.where(x > 0, x, 0)

def relu_prime(act):
    return np.where(act > 0, 1, 0)

NON_LINEAR_FUNCTIONS = {
    'sigmoid': sigmoid,
    'relu': relu,
    'linear': lambda x: x
}

NON_LINEAR_DERIVATIVES = {
    'sigmoid': sigmoid_prime,
    'relu': relu_prime,
    'linear': lambda act: 1
}

class Layer:

    def __init__(self, units, name, gate):
        self.units = units
        self.name = name
        
        if gate not in NON_LINEAR_FUNCTIONS:
            raise ValueError(f"Please select from {list(NON_LINEAR_FUNCTIONS.keys())}. {gate} is invalid")
        
        self.gate = gate

    def init_params(self, incoming_units: int):
        self.weights = (np.random.random_sample(self.units * incoming_units) - .5).reshape(-1, self.units)
        self.bias = (np.random.random_sample(self.units) - .5).reshape(-1, self.units)

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.c = np.matmul(inputs, self.weights)
        self.z = self.c + self.bias
        self.a = NON_LINEAR_FUNCTIONS[self.gate](self.z)
        return self.a
    
    def backward_pass(self, d_j_d_a, learning_rate):      
        batch_size = self.inputs.shape[0]

        d_a_d_z = NON_LINEAR_DERIVATIVES[self.gate](self.a)
        d_z_d_c = 1 

        d_j_d_c = d_j_d_a * d_a_d_z * d_z_d_c

        d_j_d_w = np.matmul(self.inputs.T, d_j_d_c) / batch_size
        self.weights -= learning_rate * d_j_d_w

        d_j_d_b = np.sum(d_j_d_c, axis=0, keepdims=True) / batch_size
        self.bias -= learning_rate * d_j_d_b

        d_j_d_a_0 = np.matmul(d_j_d_c, self.weights.T) / batch_size        
        return d_j_d_a_0
    

def binary_cross_entropy(out, y_batch):
    return np.mean(-y * np.log(out) - (1 - y_batch) * np.log(1 - out))

def binary_cross_entropy_prime(out, y_batch):
    return (out - y_batch) / (out * (1 - out)) 

def mean_squared_error(out, y_batch):
    return (1 / 2 * out.shape[0]) * np.sum((out - y_batch)**2)

def mean_squared_error_prime(out, y_batch):
    return out - y_batch

LOSS_FUNCS = {
    'binary_cross_entropy': {
        'loss': binary_cross_entropy,
        'prime': binary_cross_entropy_prime
    },
    'mean_squared_error': {
        'loss': mean_squared_error,
        'prime': mean_squared_error_prime
    }
}
    
class Sequential:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.weights_initalized = False
        self.compiled = False

    def compile(self, loss_func):
        if loss_func not in LOSS_FUNCS:
            raise ValueError(f'Loss function {loss} not in {list(LOSS_FUNCS.keys())} ')
        
        self.loss_func = loss_func
        self.compiled = True

    def predict(self, input):
        if not self.weights_initalized:
            self.initalize_weights(input.shape[1])
            self.weights_initalized = True

        a = input
        for layer in self.layers:
            a = layer.forward_pass(a)
            # We clip here because if 1 or 0, then we get weird rounding 
            # errors eventually. This is a common trick in ML.
            a = np.clip(a, 1e-10, 1 - 1e-10) 
        return a
    
    def initalize_weights(self, input_len):
        num_units = input_len
        for layer in self.layers:
            layer.init_params(num_units)
            num_units = layer.units
    
    def fit(self, x, y, epochs=1, batch_size=32, learning_rate=0.007):

        if not self.compiled:
            raise RuntimeError(f'Please compile before fitting')

        if any(np.isnan(x)):
            raise ValueError(f'Training data x contains nan values. Not valid')
        elif any(np.isnan(y)):
            raise ValueError(f'Training data y contains nan values. Not valid')
        
        loss_func = LOSS_FUNCS[self.loss_func]['loss']
        loss_func_prime = LOSS_FUNCS[self.loss_func]['prime']

        losses = []
        loss = loss_func(self.predict(x), y)
        print(f"Epoch {0}, Loss: {loss}")
        losses.append(loss)
        for epoch in range(epochs):
            print(f'Staring epoch {epoch}')
            for i in range(0, len(x), batch_size):
                x_batch = x[i:min(i+batch_size, len(x) - 1)]
                y_batch = y[i:min(i+batch_size, len(x) - 1)]
                
                out = self.predict(x_batch)
                d_j_d_a = loss_func_prime(out, y_batch)

                for layer in reversed(self.layers):
                    d_j_d_a = layer.backward_pass(d_j_d_a, learning_rate)
                    
            # Optional: print loss after each epoch
            loss = loss_func(self.predict(x), y)
            print(f"Epoch {epoch + 1}, Loss: {loss}")
            losses.append(loss)

        # Make a plot of the loss, and display it
        import matplotlib.pyplot as plt
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

#x = np.array([[1], [2], [1], [2], [1], [2], [1], [2]])
#y = np.array([[0], [1], [0], [1], [0], [1], [0], [1]])
x = np.arange(0, 100).reshape(-1, 1)
y = x + 1

nn = Sequential([
    Layer(10, 'L1', 'sigmoid'),
    Layer(24, 'L2', 'sigmoid'),
    Layer(1, 'L3', 'linear'),
])

nn.compile('mean_squared_error')
nn.fit(x, y, epochs=100)

# TODO:
# 1. Fix it for log
# 2. Add regularization parameter