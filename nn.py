import numpy as np
import math

"""
Activation/loss functions and derivatives
"""


# softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def derivative_softmax_input(a, lab):
    # derivative of error with respect to softmax input is the one hot encoded label vector subtracted from
    # the softmax output. Crazy math going on.
    return np.subtract(a, lab)


# relu
def relu(n):
    return max(0, n)


def relu_derivative(n):
    return (n > 0) * 1


def multi_class_cross_entropy(a, lab):
    # 1 * value from output vector at same index as class in the one hot encoding
    return math.log(a.A1[np.argmax(lab.A1)])


# vectorize functions that are applied to np vector/array/matrix
v_relu = np.vectorize(relu)
v_relu_d = np.vectorize(relu_derivative)
v_soft_max = np.vectorize(softmax)

"""
Importing data
"""

# get training inputs and labels.
training_data = np.loadtxt("data/mnist_train.csv", delimiter=",")
test_data = np.loadtxt("data/mnist_test.csv", delimiter=",")

# separate data and labels
training_labels = np.copy(training_data[:, 0])
test_labels = np.copy(test_data[:, 0])

training_inputs = np.copy(training_data[:, 1:])
test_inputs = np.copy(test_data[:, 1:])

# normalize training and test inputs. Could be more dynamic and use training_inputs.max or something.
training_inputs = np.matrix(np.divide(training_inputs, 255))
test_inputs = np.matrix(np.divide(test_inputs, 255))

# encode classes for easy error calculation.
# NOTE: know there is 10 classes 0-9. Maybe should find dynamically
encoded_training_labels = np.zeros((len(training_data), 10))
encoded_test_labels = np.zeros((len(test_data), 10))

print(np.shape(encoded_training_labels))

for i in range(len(training_data)):
    # set the element corresponding to each input and its classification to 1.
    encoded_training_labels[i][int(training_labels[i])] = 1

for i in range(len(test_data)):
    encoded_test_labels[i][int(test_labels[i])] = 1

np.matrix(encoded_training_labels)
np.matrix(encoded_test_labels)

"""
Hyper parameters
"""
LEARNING_RATE = .001
input_count, layer_one_node_count = np.shape(training_inputs)  # layer one node count == feature count
layer_two_node_count = 200  # second layer nodes
layer_three_node_count = 100  # third layer nodes
layer_four_node_count = 10  # layer four node count == class count
epochs = 10

"""
initialize weights & bias
"""
weights = [
    # weights matrices for Li->Lj should be dim (m, n) where m = count(Lj nodes) and n = count(Li nodes)
    # e.g here L1->L1 weights matrix has dims (50, 784). So input vector (784, 1) can be multiplied to produce (50, 1).
    np.matrix(np.random.uniform(-.1, .1, size=(layer_two_node_count, layer_one_node_count))),
    np.matrix(np.random.uniform(-.1, .1, size=(layer_three_node_count, layer_two_node_count))),
    np.matrix(np.random.uniform(-.1, .1, size=(layer_four_node_count, layer_three_node_count)))
]
# layers * nodes_in_current_layer * 1
biases = [
    np.matrix(np.random.uniform(-.1, .1, size=(layer_two_node_count, 1))),
    np.matrix(np.random.uniform(-.1, .1, size=(layer_three_node_count, 1))),
    np.matrix(np.random.uniform(-.1, .1, size=(layer_four_node_count, 1)))
]

epoch = 0
# training loop
while epoch < epochs:
    correct = 1
    # for i in range(1):
    for i in range(len(training_inputs)):

        # get input and label
        input = np.matrix(training_inputs[i]).T
        label = np.matrix(encoded_training_labels[i]).T

        # forward pass

        # L1->L2 relu(W*input + bias)
        L2_z = (weights[0] * input) + biases[0]
        # apply relu
        L2_a = v_relu(L2_z)

        # L2->L3 relu(W*L1_a + bias)
        L3_z = (weights[1] * L2_a) + biases[1]
        L3_a = v_relu(L3_z)

        # L3->L4 softmax(W*L3_a + bias)
        L4_z = (weights[2] * L3_a) + biases[2]
        L4_a = softmax(L4_z)  # network output.

        if np.argmax(L4_a) == np.argmax(label):
            correct += 1

        # get network error
        if L4_a[np.argmax(L4_a)] == np.NAN:
            continue
        else:
            error = multi_class_cross_entropy(L4_a, label)

        # Back propagate
        # will store weight and bias deltas in here for now
        weight_deltas = []
        bias_deltas = []

        # derivative of error wrt L3->L4 affine
        # dE/dZ[i] = a[i] - label[i] (Can prove and work through from scratch another time. This simplification will be
        # more efficient anyway).
        layer_four_error = derivative_softmax_input(L4_a, label) * error
        # (10,1) * (1,25) = (10, 25) (same dimension as weights L3->L4)
        weight_deltas.insert(0, (layer_four_error * L3_a.T) * LEARNING_RATE)
        bias_deltas.insert(0, layer_four_error * LEARNING_RATE)

        # propagate error to layer 3. (pretty much want dLayer_four_error/dZ_3)
        # L3->L4 W^t * Layer four error and then element wise product with relu derivative for layer 3.
        layer_three_error = np.multiply((weights[2].T * layer_four_error), relu_derivative(L3_a))
        # L2->3 weight deltas.
        weight_deltas.insert(0, (layer_three_error * L2_a.T) * LEARNING_RATE)
        bias_deltas.insert(0, layer_three_error * LEARNING_RATE)

        # propagate layer 3 error back to layer 2
        layer_two_error = np.multiply((weights[1].T * layer_three_error), relu_derivative(L2_a))
        # L1->L2 weight deltas
        weight_deltas.insert(0, (layer_two_error * input.T) * LEARNING_RATE)
        bias_deltas.insert(0, layer_two_error * LEARNING_RATE)

        # element wise sum of weight/bias deltas and weights/biases
        for w in range(len(weights)):
            weights[w] = weights[w] + weight_deltas[w]
            biases[w] = biases[w] + bias_deltas[w]

        print("Epoch: ", epoch, "input: ", i + 1, "  Accuracy: ", correct / (i + 1), "  Error: ", error)
        # FORWARD PASS record outputs for finding derivative of error of nodes in each layer with respect to weights
        # feeding into layer.

    epoch += 1

# validate on test set
test_count = 0
test_correct = 0
for t in range(len(test_inputs)):
    # forward pass
    input = np.matrix(test_inputs[t]).T
    label = np.matrix(encoded_test_labels[t]).T

    L2_z = (weights[0] * input) + biases[0]
    L2_a = v_relu(L2_z)

    L3_z = (weights[1] * L2_a) + biases[1]
    L3_a = v_relu(L3_z)

    L4_z = (weights[2] * L3_a) + biases[2]
    L4_a = softmax(L4_z)

    error = multi_class_cross_entropy(L4_a, label)

    if np.argmax(L4_a.A1) == np.argmax(label.A1):
        test_correct += 1

    test_count += 1

    print("Test input: ", test_count, " Accuracy: ", (test_correct / test_count), "  Error: ", error)

"""
To do now:

- make all hard coded number/ parameters into vars for easy changing
- make network capable of batches.
- would network train faster if activation and loss was calc'd without calling function? e.g calc activation in-line
"""
