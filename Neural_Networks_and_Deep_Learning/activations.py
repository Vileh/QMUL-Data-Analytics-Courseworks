import numpy as np

# Rectified Linear Unit (ReLU) activation function
class ReLU():
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
    
    def backward(self, output_grad, learning_rate):
        return np.multiply(output_grad, np.where(self.input <= 0, 0, 1))

# Exponential Linear Unit (ELU) activation function
class ELU():
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
    
    def backward(self, output_grad, learning_rate):
        return np.multiply(output_grad, np.where(self.input > 0, 1, self.alpha * np.exp(self.input)))
    
# Softmax activation function
class Softmax():
    def forward(self, input):
        # Subtract the maximum value from each row to avoid numerical instability
        input_max = np.max(input, axis=1, keepdims=True)
        input -= input_max

        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        eye = np.repeat(np.eye(output_gradient.shape[1]), output_gradient.shape[0], axis=0)
        input_grad = (self.output[:, :, None] * (eye - self.output[:, None, :]))
        
        return (output_gradient[:, None, :] @ input_grad).reshape(output_gradient.shape)