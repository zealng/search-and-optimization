# Gradient Descent, Newton Descent, Conjugate Descent
import numpy as np
from typing import *


class GradientDescent:
    def __init__(self, func: Callable, grad: Callable, lr: float, num_iters: int, threshold: float = 0.):
        """
        Initialize gradient descent.
        :param func: Function to minimize, should accept an np.ndarray and return a scalar.
        :param grad: Function that returns the gradient of func, should accept an np.ndarray and return an np.ndarray.
        :param lr: Learning rate.
        :param num_iters: Number of iterations to perform.
        :param threshold: Threshold for convergence and early stopping.
        """
        assert callable(func)
        assert callable(grad)
        assert lr > 0
        assert num_iters >= 0
        assert threshold >= 0
        self.f = func
        self.g = grad
        self.lr = lr
        self.num_iters = num_iters
        self.threshold = threshold

        # Data for plotting
        self.parameters = []
        self.function_values = []

    def optimize(self, init: np.ndarray):
        """
        Run gradient descent.
        :param init: Initial value for starting gradient descent. Should be able to be called with func and grad.
        """
        x = init
        prev_f_x = None
        for _ in range(self.num_iters):
            self.parameters.append(x)

            # Get the function value
            f_x = self.f(x)
            self.function_values.append(f_x)

            # Check for convergence
            if prev_f_x is not None and np.linalg.norm(prev_f_x - x) < self.threshold:
                break

            # Make a step towards the optimal solution
            x -= self.lr * self.g(x)

            # Prepare for the next iteration
            prev_f_x = f_x


class NewtonDescent:
    def __init__(self, func: Callable, grad: Callable, hess: Callable, num_iters: int, threshold: float = 0.):
        """
        Initialize newton descent.
        :param func: Function to minimize, should accept an np.ndarray and return a scalar.
        :param grad: Function that returns the gradient of func, should accept an np.ndarray and return an np.ndarray.
        :param hess: Function that returns the hessian of func, should accept an np.ndarray and return an np.ndarray.
        :param num_iters: Number of iterations to perform.
        :param threshold: Threshold for convergence and early stopping.
        """
        assert callable(func)
        assert callable(grad)
        assert callable(hess)
        assert num_iters >= 0
        assert threshold >= 0
        self.f = func
        self.g = grad
        self.h = hess
        self.num_iters = num_iters
        self.threshold = threshold

        # Data for plotting
        self.parameters = []
        self.function_values = []

    def optimize(self, init: np.ndarray):
        """
        Run newton descent.
        :param init: Initial value for starting newton descent. Should be able to be called with func, grad, and hess.
        """
        x = init
        prev_f_x = None
        for _ in range(self.num_iters):
            self.parameters.append(x)

            # Get the function value
            f_x = self.f(x)
            self.function_values.append(f_x)

            # Check for convergence
            if prev_f_x is not None and np.linalg.norm(prev_f_x - x) < self.threshold:
                break

            # Make a step towards the optimal solution
            x -= np.linalg.inv(self.h(x)) @ self.g(x)

            # Prepare for the next iteration
            prev_f_x = f_x


class ConjugateGradientDescent:
    def __init__(self, A: np.ndarray, b: np.ndarray, loss: Callable, lr: float, num_iters: int, threshold: float = 0.):
        """
        Initialize conjugate gradient descent, which solves the problem Ax = b.
        Implementation based on https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
        :param A: A, must be symmetric positive-definite.
        :param b: b
        :param loss: Optional function that converts A and b into a loss value. Called as func(A, x, b).
        :param lr: Learning rate.
        :param num_iters: Number of iterations to perform.
        :param threshold: Threshold for convergence and early stopping.
        """
        assert A == A.T  # Symmetric
        assert all(np.linalg.eigvals(A) > 0)  # PSD
        assert callable(loss)
        assert lr > 0
        assert num_iters >= 0
        assert threshold >= 0
        self.A = A
        self.b = b
        self.loss = loss
        self.lr = lr
        self.num_iters = num_iters
        self.epsilon = threshold

        # Data for plotting
        self.parameters = []
        self.loss_values = []

    def optimize(self, init: np.ndarray):
        """
        Run conjugate gradient descent.
        :param init: Initial value for starting conjugate gradient descent. Should be able to be called with func and
            grad.
        """
        x = init
        r = self.b - (self.A @ x)
        d = r
        delta_new = r.T @ r
        delta_0 = delta_new
        for i in range(self.num_iters):
            # Get the loss value
            self.parameters.append(x)
            loss = self.loss(self.A, x, self.b)
            self.loss_values.append(loss)

            # Check for convergence
            if delta_new < (self.epsilon ** 2) * delta_0:
                break

            # Make a step towards the optimal solution
            q = self.A @ d
            alpha = delta_new / (d.T @ q)
            x = x + (alpha * d)

            # Refresh r to avoid floating point error
            if i % 50 == 0:
                r = self.b - (self.A @ x)
            else:
                r = r - (alpha * q)

            # Prepare for the next iteration
            delta_old = delta_new
            delta_new = r.T @ r
            beta = delta_new / delta_old
            d = r + (beta * d)





