from typing import List
import numpy as np

def sigmoid(x): 
    return 1 / (1 + np.exp(-x)) 

def relu(x):
    return np.where(x > 0, x, 0)

class Layer:

    def __init__(self, units, name):
        self.units = units
        self.name = name

    def init_params(self, incoming_units: int):
        self.weights = (np.random.random_sample(self.units * incoming_units) - .5).reshape(-1, self.units)
        self.bias = (np.random.random_sample(self.units) - .5).reshape(-1, self.units)

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.c = np.matmul(inputs, self.weights)
        self.z = self.c + self.bias
        self.a = sigmoid(self.z)
        return self.a
    
    def backward_pass(self, d_j_d_a, learning_rate):      
        batch_size = self.inputs.shape[0]

        d_a_d_z = self.a * (1 - self.a)
        d_z_d_c = 1 

        d_j_d_c = d_j_d_a * d_a_d_z * d_z_d_c

        d_j_d_w = np.matmul(self.inputs.T, d_j_d_c) / batch_size
        self.weights -= learning_rate * d_j_d_w

        d_j_d_b = np.sum(d_j_d_c, axis=0, keepdims=True) / batch_size
        self.bias -= learning_rate * d_j_d_b

        d_j_d_a_0 = np.matmul(d_j_d_c, self.weights.T) / batch_size        
        return d_j_d_a_0

    
class Sequential:

    def __init__(self, layers: List[Layer]):
        self.layers = layers
        self.compiled = False

    def predict(self, input):
        if not self.compiled:
            self.compile(input.shape[1])
            self.compiled = True

        a = input
        for layer in self.layers:
            a = layer.forward_pass(a)
            # We clip here because if 1 or 0, then we get weird rounding 
            # errors eventually. This is a common trick in ML.
            a = np.clip(a, 1e-10, 1 - 1e-10) 
        return a
    
    def compile(self, input_len):
        num_units = input_len
        for layer in self.layers:
            layer.init_params(num_units)
            num_units = layer.units
    
    def fit(self, x, y, epochs=1, batch_size=32, learning_rate=0.007):

        if any(np.isnan(x)):
            raise ValueError(f'Training data x contains nan values. Not valid')
        elif any(np.isnan(y)):
            raise ValueError(f'Training data y contains nan values. Not valid')

        losses = []
        for epoch in range(epochs):
            print(f'Staring epoch {epoch}')
            for i in range(0, len(x), batch_size):
                x_batch = x[i:min(i+batch_size, len(x) - 1)]
                y_batch = y[i:min(i+batch_size, len(x) - 1)]
                
                out = self.predict(x_batch)
                d_j_d_a = (out - y_batch) / (out * (1 - out)) 

                for layer in reversed(self.layers):
                    d_j_d_a = layer.backward_pass(d_j_d_a, learning_rate)
                    
            # Optional: print loss after each epoch
            loss = np.mean(-y * np.log(self.predict(x)) - (1 - y) * np.log(1 - self.predict(x)))
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
x = np.arange(-4000, 4000).reshape(-1, 1)
y = np.where(x > 0, 1, 0)

nn = Sequential([
    Layer(10, 'L1'),
    Layer(24, 'L2'),
    Layer(1, 'L3'),
])

loss = np.mean(-y * np.log(nn.predict(x)) - (1 - y) * np.log(1 - nn.predict(x)))
print(f"Epoch {0}, Loss: {loss}")
nn.fit(x, y, epochs=100)