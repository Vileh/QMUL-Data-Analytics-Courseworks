import numpy as np
from scipy import signal


# Convolutional layer
class ConvolutionalLayer():
    def __init__(self, input_shape:tuple, output_depth:int, kernel_size, padding=0, stride=1, bias:bool=True):
        self.input_depth, input_height, input_width = input_shape
        self.output_depth = output_depth
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.bias = bias

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        if isinstance(stride, int):
            self.stride = (stride, stride)

        self.kernels_shape = (output_depth, self.input_depth, self.kernel_size[0], self.kernel_size[1])
        self.output_shape = (output_depth, int((input_height - self.kernel_size[0] + 2*self.padding[0] + self.stride[0]) / self.stride[0]), int((input_width - self.kernel_size[1] + 2*self.padding[1] + self.stride[0]) / self.stride[0]))
        self.dilated_gradient_shape = (output_depth, (input_height - self.kernel_size[0] + 2*self.padding[0] + 1), (input_width - self.kernel_size[1] + 2*self.padding[1] + 1))

        # Initialise weights and biases
        self.weights = np.random.randn(output_depth, self.input_depth, self.kernel_size[0], self.kernel_size[1]) * np.sqrt(2 / (self.input_depth * self.kernel_size[0] * self.kernel_size[1]))
        self.biases = np.zeros(self.output_depth)

    def forward(self, input):     
        # Padding
        input = np.pad(input, ((0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])))
        self.input = input

        # Calculate output
        self.output = np.zeros(self.output_shape) + self.biases[:, np.newaxis, np.newaxis]
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                # Cross correlation
                self.output[i] += signal.correlate2d(input[j], self.weights[i, j], "valid")[::self.stride[0], ::self.stride[1]]
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input.shape)

        # Initialise dilated output gradient matrix
        dilated_output_gradient = np.zeros(self.dilated_gradient_shape)

        for i in range(self.output_depth):
            
            # Make the dilated output gradient
            for index_x, row in enumerate(output_gradient[i]):
                for index_y, gradient in enumerate(row):
                    dilated_output_gradient[i, index_x * (self.stride[0]), index_y * (self.stride[1])] = gradient

            # Calculate the gradients
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], dilated_output_gradient[i], "valid")[::self.stride[0], ::self.stride[1]]
                input_gradient[j] += signal.convolve2d(dilated_output_gradient[i], self.weights[i, j], "full")

        # Update the weights
        self.weights -= learning_rate * kernels_gradient

        # Update the biases if needed
        if self.bias:
            self.biases -= learning_rate * np.sum(output_gradient, axis=(1, 2))

        # Crop the padding gradients
        if self.padding[0] != 0:
            input_gradient = input_gradient[:, self.padding[0]:-self.padding[0], :]
        if self.padding[1] != 0:
            input_gradient = input_gradient[:, :, self.padding[1]:-self.padding[1]]

        return input_gradient
    

# Max pooling layer
class MaxPooling():
    def __init__(self, input_shape, kernel_size, padding, stride):
        self.input_depth, input_height, input_width = input_shape
        self.output_depth = self.input_depth
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        if isinstance(padding, int):
            self.padding = (padding, padding)
        if isinstance(stride, int):
            self.stride = (stride, stride)

        self.output_shape = (self.output_depth, int((input_height - self.kernel_size[0] + 2*self.padding[0] + self.stride[0]) / self.stride[0]), int((input_width - self.kernel_size[1] + 2*self.padding[1] + self.stride[0]) / self.stride[0]))

    def forward(self, input):      
        # Padding
        input = np.pad(input, ((0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), constant_values=-np.inf)
        self.input = input

        # Calculate output
        self.output = np.zeros(self.output_shape)
        for i in range(self.output_depth):
            for x in range(self.output_shape[1]):
                for y in range(self.output_shape[2]):
                    self.output[i, x, y] = np.max(input[i, x*self.stride[0]:x*self.stride[0]+self.kernel_size[0], y*self.stride[1]:y*self.stride[1]+self.kernel_size[1]])

        return self.output
    
    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input.shape)
        
        for i in range(self.output_depth):
            for x in range(self.output_shape[1]):
                for y in range(self.output_shape[2]):
                    window = self.input[i, x*self.stride[0]:x*self.stride[0]+self.kernel_size[0], y*self.stride[1]:y*self.stride[1]+self.kernel_size[1]]
                    max_value = np.max(window)
                    input_gradient[i, x*self.stride[0]:x*self.stride[0]+self.kernel_size[0], y*self.stride[1]:y*self.stride[1]+self.kernel_size[1]] += (window == max_value) * output_gradient[i, x, y]
        
        # Crop the padding gradients
        if self.padding[0] != 0:
            input_gradient = input_gradient[:, self.padding[0]:-self.padding[0], :]
        if self.padding[1] != 0:
            input_gradient = input_gradient[:, :, self.padding[1]:-self.padding[1]]

        return input_gradient
    

# Flatten (Reshape) layer
class Reshape():
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
    

# Fully connected layer
class Dense():
    def __init__(self, input_size, output_size, bias:bool=True):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / np.sum(input_size))
        self.bias = np.zeros(output_size) if bias else None

    def forward(self, input):
        self.input = input
        output = np.dot(input, self.weights)
        if self.bias is not None:
            output += self.bias
        return output

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient.T, self.input).T
        input_gradient = np.dot(self.weights, output_gradient.T).T

        self.weights -= learning_rate * weights_gradient

        if self.bias is not None:
            self.bias -= learning_rate * output_gradient.sum(axis=1)

        return input_gradient
    

# Instance Normalisation layer
class InstanceNorm():
    def __init__(self, num_features, eps=1e-5, affine=True):
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.gamma = np.ones((num_features, 1))
            self.beta = np.zeros((num_features, 1))
        else:
            self.gamma = None
            self.beta = None
        
        self.mean = None
        self.std = None
        self.x_flat = None

    def forward(self, x):
        self.input = x

        # Reshape the input tensor to compute mean and standard deviation along the feature dimension
        self.x_flat = x.reshape(self.num_features, -1)
        
        # Compute the mean and standard deviation along the feature dimension
        self.mean = np.mean(self.x_flat, axis=-1, keepdims=True)
        self.std = np.std(self.x_flat, axis=-1, keepdims=True)
        
        # Normalize the input tensor
        self.y_flat = (self.x_flat - self.mean) / (self.std + self.eps)
        
        # Apply the learnable scale and shift parameters if affine is True
        if self.affine:
            self.y_flat = self.gamma * self.y_flat + self.beta
        
        # Reshape the normalized output tensor to the original shape
        self.y = self.y_flat.reshape(x.shape)
        
        return self.y

    def backward(self, output_gradient, learning_rate):
        # Reshape the output gradient tensor
        dy_flat = output_gradient.reshape(self.num_features, -1)

        n = dy_flat.shape[-1]
        self.mean = self.mean[:, np.newaxis]
        self.std = self.std[:, np.newaxis]
        
        # Compute the gradients of the scale and shift parameters
        if self.affine:
            dgamma = np.array([np.dot(self.y_flat[i], dy_flat[i]) for i in range(len(self.y_flat))])
            dbeta = np.sum(dy_flat, axis=-1)
        else:
            dgamma = None
            dbeta = None

        delta = np.eye(n)[None, :, :]
        delta = np.broadcast_to(delta, (self.num_features, n, n))

        term1 = (delta - 1 / n) / (self.std + self.eps)
        term2 = ((self.input - self.mean) / (n * self.std)).reshape(self.num_features, -1)[:, :, np.newaxis]
        term3 = ((self.input - self.mean) / ((self.std + self.eps)**2)).reshape(self.num_features, -1)[:, np.newaxis, :]

        dy_dx = term1 - term2 * term3

        dx_flat = np.array([np.dot(dy_dx[i], dy_flat[i]) for i in range(len(dy_dx))])

        if self.affine:
            dx_flat *= self.gamma
        
        # Reshape the input gradient tensor to the original shape
        dx = dx_flat.reshape(output_gradient.shape)

        # Update the parameters
        if self.affine:
            self.gamma -= learning_rate * dgamma[:, np.newaxis]
            self.beta -= learning_rate * dbeta[:, np.newaxis]
        
        return dx