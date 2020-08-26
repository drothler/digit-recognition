from LearningProcess import weights, training_vectors, testing_vectors, hidden_vectors, layerAmount
from DigitConvert import training_labels, test_labels
import numpy as np

neurons = []

outputs = []


def maximum_function(value) -> float:
    return np.maximum(value, 0)


def softmax_function(array) -> float:
    return np.exp(array)/sum(np.exp(array))


def process_pixel_neurons(index):
    first_vector = maximum_function(weights[0].dot(training_vectors[index]))
    neurons.append(first_vector)


def process_hidden_vectors():
    for step in range(layerAmount-1):
        # we take the previous n-th vector from "neurons" and multiply it with the n+1-th weight matrix
        # to get the n+1-th vector to store in "neurons"
        neurons.append(maximum_function(weights[step+1].dot(neurons[step])))


def process_output_vector():
    global outputs
    outputs.append(softmax_function(weights[layerAmount].dot(neurons[layerAmount-1])))


def process_vectors(index):
    process_pixel_neurons(index)
    process_hidden_vectors()
    process_output_vector()


def backpropagate(index):
    return


def iterate_training_samples():
    for img in range(len(training_vectors)):
        process_vectors(img)
        backpropagate(img)


def neuron_cost_function(index, position):
    p = np.zeros(10)
    p[index] = 1
    return 0.5 * np.square(p - outputs[position])


process_vectors(0)
print(outputs[0].argmax())
print(outputs)
print(neuron_cost_function(training_labels[0], 0))