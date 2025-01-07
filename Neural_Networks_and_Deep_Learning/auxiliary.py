import numpy as np
import time

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, x_test, y_test, epochs = 1000, learning_rate = 0.01, lr_decay = 0.99, batch_size = 32, verbose = True, test_epochs=10):
    errors = []
    test_scores = []
    for e in range(epochs):
        error = 0
        indices = np.random.permutation(len(x_train))[:batch_size]            
        for x, y in zip(x_train[indices], y_train[indices]):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= batch_size
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}, lr = {learning_rate}")

        if e % test_epochs == 0:
            correct = 0
            indices = np.random.permutation(len(x_test))[:100]      
            for x, y in zip(x_test[indices], y_test[indices]):
                output = predict(network, x)
                correct += np.argmax(output) == np.argmax(y)

            test_scores.append(correct / len(x_test))

        learning_rate *= lr_decay

        errors.append(error)
    
    return errors, test_scores

def categorical_cross_entropy(y_true, y_pred):    
    # Clip predicted values to avoid log(0) errors
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred))
    
    # Average loss over samples
    loss /= y_true.shape[1]
    
    return loss

def categorical_cross_entropy_prime(y_true, y_pred):
    # Clip predicted values to avoid log(0) errors
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute gradient of cross-entropy loss
    gradient = -y_true / y_pred
    
    return gradient

def test_time(network, input, label, totals_only=False):
    output=input
    tot = 0
    for layer in network:
        start = time.monotonic()
        output = layer.forward(output)
        stop = time.monotonic()

        if not totals_only:
            print(f"\tForward {layer.__class__.__name__} time: {stop - start}")
        tot += stop - start

    print(f"Forward total: {tot}")

    y = label
    grad = categorical_cross_entropy_prime(y, output)
    tot = 0
    for layer in reversed(network):
        start = time.monotonic()
        grad = layer.backward(grad, 0)
        stop = time.monotonic()

        if not totals_only:
            print(f"\tBackward {layer.__class__.__name__} time: {stop - start}")
        tot += stop - start

    print(f"Backward total: {tot}")