from DigitConvert import training_images, test_images
import numpy as np


# for future adaptability, different types of neural networks
# 0 = Explicit approach
# 1 = stochastic approach -> use different subsets to go through training phases
network_type = 0

# for future adaptability, different types of neuron functions
function_type = 0

# specifies the amount of layers of neurons in the neural network
layerAmount = 2

# specifies the amount of neurons per layer
neuronsPerLayer = 6

# specifies image size
pixels = 28

# weights of edges
weights = []

# vectors storing the data for each input picture
training_vectors = []

# vectors storing the data for each testing picture
testing_vectors = []

# vectors storing the data for the hidden layers
hidden_vectors = []




def init_vectors() -> None:
    global training_vectors
    global testing_vectors
    for index in range(len(test_images)):
        temp1 = []
        temp2 = []
        for value in range(len(training_images[index])):
            temp1.append(training_images[index][value] / 255)
            temp2.append(test_images[index][value] / 255)
        training_vectors.append(temp1)
        testing_vectors.append(temp2)
    for index in range(len(test_images), len(training_images)):
        temp = []
        for value in range(len(training_images[index])):
            temp.append(training_images[index][value]/255)
        training_vectors.append(temp)


def generate_weight_matrices() -> None:
    global weights

    # in case there are zero hidden layers, we only need weights from our pixels to the result neurons
    if layerAmount == 0:
        matrix = np.random.randn(10, pixels*pixels)
        weights.append(matrix)
    else:
        # if there are more then zero hidden layers, we have to create a specific weight matrix for pixels to neurons
        # and a specific matrix from hidden neurons to output

        # start to hidden layer
        weights.append(np.random.randn(neuronsPerLayer, pixels*pixels))

        # hidden layer to hidden layer
        for layer in range(layerAmount-1):
            temp_weight_matrix = np.random.randn(neuronsPerLayer, neuronsPerLayer)
            weights.append(temp_weight_matrix)

        # hidden layer to output
        weights.append(np.random.randn(10, neuronsPerLayer))


def generate_hidden_vectors() -> None:
    for index in range(layerAmount):
        hidden_vectors.append(np.zeros(neuronsPerLayer))


# initializing all necessary data to start off the learning process
init_vectors()
generate_weight_matrices()
generate_hidden_vectors()


