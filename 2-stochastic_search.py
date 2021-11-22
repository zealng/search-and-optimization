# Simulated Annealing, Cross-Entropy Methods, Search Gradient
import numpy as np
from typing import *


class SimulatedAnnealing:
    """
    Simulated annealing method for function minimization. Uses np.random.multivariate_normal as the sampling distribution.
    """
    def __init__(self, func: Callable, num_iters: int, cooling_func: Optional[Union[Callable, str]], init_temp: float = 100):
        """
        Initialize simulated annealing. Uses np.random.multivariate_normal as the sampling distribution.
        :param func: Function to minimize, should accept an np.ndarray and return a scalar.
        :param num_iters: Number of iterations to perform.
        :param cooling_func: A positive function that decreases over time, as a function of the current iteration
            number. Default: "fast" annealing. Also available are "exponential" and "log" annealing.
        :param init_temp: Initial temperature.
        """
        assert callable(func)
        assert num_iters >= 0
        assert callable(cooling_func) or cooling_func is None or type(cooling_func) == str
        self.f = func
        self.num_iters = num_iters

        if cooling_func is None or cooling_func == 'fast':
            self.temp = lambda k: init_temp / k
        elif cooling_func == 'exponential':
            self.temp = lambda k: init_temp * (0.5 ** k)
        elif cooling_func == 'log':
            self.temp = lambda k: init_temp * np.log(2) / np.log(k + 1)
        else:
            self.temp = cooling_func

        # Data for plotting
        self.samples = []
        self.function_values = []

    def optimize(self, init_x: np.ndarray, init_cov: np.ndarray, init_temp: float):
        """
        Run simulated annealing.
        :param init_x: Initial parameter value.
        :param init_cov: Initial covariance value.
        """
        x = init_x
        cov = init_cov
        for k in range(self.num_iters):
            # Get the current temperature
            T = self.temp(k)

            # Get the function value
            self.samples.append(x)
            fx = self.f(x)
            self.function_values.append(fx)

            # Sample a step
            x_new = np.random.multivariate_normal(x, cov)
            fx_new = self.f(x_new)

            # Accept or reject the step
            if fx_new >= fx:
                if np.random.rand(1) < np.exp((fx - fx_new) / T):
                    x = x_new
            else:
                x = x_new


class CrossEntropy:
    """
    Cross-entropy method for function minimization. Uses np.random.multivariate_normal as the sampling distribution.
    """

    def __init__(self, func: Callable, num_iters: int, sample_size: int, elite_size: int):
        """
        Initialize cross entropy search.
        :param func: Function to minimize, should accept an np.ndarray and return a scalar.
        :param num_iters: Number of iterations to perform.
        :param sample_size: Number of samples to draw at each iteration.
        :param elite_size: Number of samples that should form the elite set.
        """
        assert callable(func)
        assert num_iters >= 0
        assert sample_size >= 0
        assert elite_size >= 0
        assert sample_size > elite_size
        self.f = func
        self.num_iters = num_iters
        self.sample_size = sample_size
        self.elite_size = elite_size

        # Data for plotting
        self.samples = []
        self.function_values = []

    def optimize(self, init_x: np.ndarray, init_cov: np.ndarray, cov_noise: np.ndarray = 0):
        """
        Run cross entropy search.
        :param init_x: Initial parameter value.
        :param init_cov: Initial covariance value.
        :param cov_noise: Noise that is added to the covariance matrix to avoid getting stuck in local minima.
        """
        x = init_x
        cov = init_cov
        for i in range(self.num_iters):
            # Get the function value for plotting
            self.samples.append(x)
            self.function_values.append(self.f(x))

            # Sample from current distribution
            samples = np.random.multivariate_normal(x, cov, size=self.sample_size)  # (s, 50)
            sample_values = []
            for s in samples:
                sample_values.append(self.f(s))

            # Form the elite set from the best samples
            elite_indices = sorted(range(len(sample_values)), key=lambda a: sample_values[a])[:self.elite_size]
            elite_samples = samples[elite_indices, :]

            # Update distribution based on the elite set
            x = np.mean(elite_samples, axis=0)
            cov = np.cov(elite_samples.T, bias=True) + cov_noise


class SearchGradient:
    """
    Search gradient method for function minimization. Uses np.random.multivariate_normal as the sampling distribution.
    """

    def __init__(self, func: Callable, num_iters: int, sample_size: int, lr: float):
        """
        Initialize search gradient.
        :param func: Function to minimize, should accept an np.ndarray and return a scalar.
        :param num_iters: Number of iterations to perform.
        :param sample_size: Number of samples to draw at each iteration.
        :param lr: Learning rate.
        """
        assert callable(func)
        assert num_iters >= 0
        assert sample_size >= 0
        assert lr > 0
        self.f = func
        self.num_iters = num_iters
        self.sample_size = sample_size
        self.lr = lr

        # Data for plotting
        self.samples = []
        self.function_values = []

    def optimize(self, init_x: np.ndarray, init_cov: np.ndarray, normalize_gradients: bool = True):
        """
        Run search gradient method.
        :param init_x: Initial parameter value.
        :param init_cov: Initial covariance value.
        :param normalize_gradients: Whether the gradients are normalized. Note that the gradient with respect to the
            mean and gradient with respect to the covariance are normalized separately.
        """

        N = init_x.size
        x = init_x
        cov = init_cov
        for i in range(self.num_iters):
            # Get the function value for plotting
            fx = self.f(x)
            self.function_values.append(fx)

            fitnesses = np.zeros((self.sample_size, 1))
            nabla_x_log_pis = np.zeros((self.sample_size, N))
            nabla_cov_log_pis = np.zeros((self.sample_size, N, N))
            for ki in range(self.sample_size):
                # Draw sample
                z = np.random.multivariate_normal(x.reshape(N), cov).reshape((N, 1))

                # Evaluate fitness
                fitnesses[ki, :] = self.f(z)

                # Calculate log-derivatives
                diff = z - x
                cov_inv = np.linalg.inv(cov)
                nabla_x_log_pis[ki, :] = (cov_inv @ diff).reshape(N)
                nabla_cov_log_pis[ki, ...] = (- 0.5 * cov_inv) + (0.5 * cov_inv @ diff @ diff.T @ cov_inv)

            # Update mean and covariance
            nabla_x_J = (1. / self.sample_size) * np.sum(nabla_x_log_pis * fitnesses, axis=0).reshape(N, 1)
            nabla_cov_J = (1. / self.sample_size) * np.sum(nabla_cov_log_pis * fitnesses.reshape((self.sample_size, 1, 1)), axis=0)

            if normalize_gradients:
                nabla_x_J /= np.linalg.norm(nabla_x_J)
                nabla_cov_J /= np.linalg.norm(nabla_cov_J)

            x -= self.lr * nabla_x_J
            cov -= self.lr * nabla_cov_J



