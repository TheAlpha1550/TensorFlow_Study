import random
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")

# Define variables needed for plotting.
color_list = ["r-", "m-", "y-", "c-", "b-", "g-"]
color_index = 0


def perceptron(weight_vector, input_vector):
    z = np.dot(weight_vector, input_vector)
    return np.sign(z)


# First element in vector x must be 1.
# Length of w and x must be n + 1 for neuron with n inputs.
def compute_output(current_weight, current_input):
    z = 0.0

    # Compute sum of weighted inputs.
    for pair in range(len(current_weight)):
        z += current_input[pair] * current_weight[pair]

    # Apply sign (activation) function.
    if z < 0:
        return -1
    else:
        return 1


def show_learning(w):
    global color_index

    print(F"w0 = {w[0]:5.2f}\n"
          F"w1 = {w[1]:5.2f}\n"
          F"w2 = {w[2]:5.2f}\n")

    if color_index == 0:
        # NAND
        plt.plot([1.0], [1.0], "b_", markersize = 12)
        plt.plot([-1.0, 1.0, -1.0], [1.0, -1.0, -1.0], "r+", markersize = 12)

        # XOR
        # plt.plot([-1.0, 1.0], [-1.0, 1.0], "b_", markersize = 12)
        # plt.plot([-1.0, 1.0], [1.0, -1.0], "r+", markersize = 12)

        plt.axis([-2, 2, -2, 2])
        plt.xlabel("x1")
        plt.ylabel("x2")

    plt_x = [-2.0, 2.0]
    if abs(w[2]) < 1e-5:
        plt_y = [-w[1] / 1e-5 * (-2.0) + (-w[0] / 1e-5), -w[1] / 1e-5 * 2.0 + (-w[0] / 1e-5)]
    else:
        plt_y = [-w[1] / w[2] * (-2.0) + (-w[0] / w[2]), -w[1] / w[2] * 2.0 + (-w[0] / w[2])]

    plt.plot(plt_x, plt_y, color_list[color_index])

    if color_index < (len(color_list) - 1):
        color_index += 1


# Define variables needed to control training processes.
random.seed(7)
LEARNING_RATE = 0.1

# Randomize order.
index_list = [0, 1, 2, 3]

# Define training examples.
# Inputs
x_train = [[1.0, -1.0, -1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [1.0, 1.0, 1.0]]

# Outputs (ground truth)
# NAND
y_train = [1.0, 1.0, 1.0, -1.0]
# XOR
# y_train = [-1.0, 1.0, 1.0, -1.0]

# Define perceptron weights.
# Initialize to some "random" variables.
random_weights = [0.2, -0.6, 0.25]

# Perceptron training loop.
all_correct = False
# while not all_correct:
for _ in range(3):

    # Randomize the order.
    random.shuffle(index_list)

    for i in index_list:
        x = x_train[i]
        y = y_train[i]

        # Perceptron function.
        p_out = perceptron(random_weights, x)

        # Update the weights if output is incorrect.
        if y != p_out:
            for j in range(len(random_weights)):
                random_weights[j] += (y * LEARNING_RATE * x[j])
            all_correct = False
            show_learning(random_weights)

plt.show()
