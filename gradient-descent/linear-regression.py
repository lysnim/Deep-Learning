
from numpy import *

def compute_error_for_given_points(b, m, points):
    # Computes the error of a given straight line:
    """
    Given some points:
    1. measure the distance of each of those points to the line
    2. Square them, and sum them
    3. Divide by the total number of points

    (1/N)*sum(1 to N: (y-(mx+b))^2 )

    We make the square because:
    1. We do not want negative values
    2. We do not care about the value, we just care about the magnitude
       so, bigger number will have even bigger squares, so we'll notice them.
    """

    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m*x + b)) **2

    return totalError/float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    # Gradient descent
    """
    Should we go up or go down?
    We compute the partial derivatives of f'm and f'b of
    (1/N) * sum(1 to N: (y-(mx+b))^2 )    and we get:

    f'm = -(2/N) * sum(1 to N: x * (y-(mx+b)))
    f'b = -(2/N) * sum(1 to N: (y-(mx+b)))
    """

    b_gradient = 0
    m_gradient = 0
    N = float(len(points))

    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
    new_m = m_current - (learning_rate * m_gradient)
    new_b = b_current - (learning_rate * b_gradient)
    return [new_m, new_b]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # Initialize
    b = starting_b
    m = starting_m

    # Change b and m in each iteration
    for i in range(num_iterations):
        [m, b] = step_gradient(b, m, array(points), learning_rate)

    return [m, b]

def main():
    points = genfromtxt("data.csv", delimiter=",")

    # Hyperparameters
    # how fast the model learns (low = too slow, high = never converge)
    learning_rate = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # Set optimal values
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)

    # Print them
    print(b)
    print(m)

if __name__ == '__main__':
    main()
