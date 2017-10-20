import numpy as np                          # math operations
from matplotlib import pyplot as plt        # plot data

def svm_sgd_plot(X, Y):
    # Initialize SVM with zeros
    w = np.zeros(len(X[0]))

    # Learning rate
    eta = 1

    # Iterations
    epochs = 100000

    # Store misclassifications
    errors = []

    for epoch in range(1, epochs):
        error = 0
        for i, x in enumerate(X):
            # Misclassification
            if (Y[i] * np.dot(X[i], w)) < 1:
                w = w + eta * (Y[i] * X[i] - 2 * (1 / epoch) * w)
                error = 1
            else:
                w = w + eta * (-2 * (1 / epoch) * w)
        errors.append(error)

    plt.plot(errors, '|')
    plt.ylim(0.5, 1.5)
    plt.axes().set_yticklabels([])
    plt.xlabel('Epoch')
    plt.ylabel('Misclassified')
    plt.show()

    return w

def main():
    # STEP 1: Define data
    # X -> Values
    X = np.array([
        [-2, 4, -1],
        [ 4, 1, -1],
        [ 1, 6, -1],
        [ 2, 4, -1],
        [ 6, 2, -1],
    ])
    # Y -> Classification
    y = np.array([-1,-1,1,1,1])

    # Plot initial database
    for d, sample in enumerate(X):
        if d < 2:
            plt.scatter(sample[0], sample[1], s = 120, marker = '_', linewidths = 2)
        else:
            plt.scatter(sample[0], sample[1], s = 120, marker = '+', linewidths = 2)

    w = svm_sgd_plot(X, y)

    x2 = [w[0], w[1], -w[1],  w[0]]
    x3 = [w[0], w[1],  w[1], -w[0]]

    x2x3 = np.array([x2, x3])
    X, Y, U, V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X, Y, U, V, scale = 1, color = 'blue')

    plt.show()


if __name__ == "__main__":
    main()
