from mnist import MNIST
import random

mndata = MNIST('samples')

# images imported as list of unsigned bytes
# labels imported as array of unsigned bytes
training_images = []
training_labels = []


# test images imported as list of unsigned bytes
# test labels imported as array of unsigned bytes
test_images = []
test_labels = []


def read_training_data() -> None:
    global training_images
    global training_labels
    training_images, training_labels = mndata.load_training()


def read_test_data() -> None:
    global test_images
    global test_labels
    test_images, test_labels = mndata.load_testing()


def print_random_number() -> None:
    index = random.randrange(0, len(training_images))
    print(mndata.display(training_images[index]))
    print(training_labels[index])


read_training_data()

read_test_data()




