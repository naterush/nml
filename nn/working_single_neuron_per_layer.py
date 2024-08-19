from typing import List
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.where(x > 0, x, 0)

class Layer:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.c = np.matmul(inputs, self.weights)
        self.z = self.c + self.bias
        self.a = sigmoid(self.z)
        return self.a
    
    def backward_pass(self, d_a, learning_rate):      
        batch_size = self.inputs.shape[0]

        d_a_d_z = self.a * (1 - self.a)
        d_c_d_w = self.inputs
        
        d_j_d_w = np.sum(d_a * d_a_d_z * d_c_d_w) / batch_size
        self.weights -= learning_rate * d_j_d_w

        d_j_d_b = np.sum(d_a * d_a_d_z) / batch_size
        self.bias -= learning_rate * d_j_d_b

        # The start of this one 
        d_c_d_a_0 = self.weights
        d_j_d_a_0 = np.sum(d_a * d_a_d_z * d_c_d_a_0) / batch_size
        
        return d_j_d_a_0

    
class Sequential:

    def __init__(self, layers: List[Layer]):
        self.layers = layers

    def predict(self, input):
        a = input
        for layer in self.layers:
            a = layer.forward_pass(a)
            # We clip here because if 1 or 0, then we get weird rounding 
            # errors eventually. This is a common trick in ML.
            a = np.clip(a, 1e-10, 1 - 1e-10) 
        return a
    
    def fit(self, x, y, epochs=1, batch_size=32, learning_rate=0.007):
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
    Layer(np.array([[.1]]), np.array([[.1]])),
    Layer(np.array([[.2]]), np.array([[.2]])),
])

loss = np.mean(-y * np.log(nn.predict(x)) - (1 - y) * np.log(1 - nn.predict(x)))
print(f"Epoch {0}, Loss: {loss}")
nn.fit(x, y, epochs=100)