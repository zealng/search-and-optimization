# Value iteration, Policy iteration
import numpy as np


def policy_eval(pi: np.ndarray, v: np.ndarray, num_states: int, T: np.ndarray, R: np.ndarray, gamma,
               threshold: float = 1e-5) -> np.ndarray:
    """
    Given certain policy probabilities, return the associated values of all states.
    :param pi: Current policy function as an array, where (action = pi[state]).
    :param v: Current value function as an array, where (value = v[state]).
    :param num_states: Number of states in the environment.
    :param T: Transition probabilities, as a 2D array.
    :param R: Rewards for certain state transitions, as a 2D array.
    :param gamma: Discount factor. 
    :param threshold: 
    :return: The new value of all states.
    """
    v_new = np.copy(v)
    v += 100

    while np.max(np.abs(v_new - v)) > threshold:
        v = np.copy(v_new)
        for s in range(num_states):
            # Deterministic policy
            a = pi[s]
            # Use in-place updating for policy evaluation
            v_new[s] = np.sum(T[s, a, :] * (R[s, a, :] + gamma * v))

    return v_new


def policy_iter(pi: np.ndarray, v: np.ndarray, num_states: int, T: np.ndarray, R: np.ndarray, gamma=1.0) -> tuple[
    np.ndarray, np.ndarray]:
    """
    Perform one iteration of the policy iteration algorithm
    :param pi: Current policy function as an array, where (action = pi[state]).
    :param v: Current value function as an array, where (value = v[state]).
    :param num_states: Number of states in the environment.
    :param T: Transition probabilities, as a 2D array.
    :param R: Rewards for certain state transitions, as a 2D array.
    :param gamma: Discount factor. 
    :return: Updated pi and v.
    """
    # Evaluate the policy
    v = policy_eval(pi=pi, v=v, num_states=num_states, T=T, R=R, gamma=gamma)

    # Improve the policy
    pi_new = np.copy(pi)
    pi += 100
    while np.any(pi_new != pi):
        pi = np.copy(pi_new)
        for s in range(num_states):
            pi_new[s] = np.argmax(np.sum(T[s, :, :] * (R[s, :, :] + gamma * v[np.newaxis, :]), axis=1))

    return pi, v


def value_iter(v: np.ndarray, num_states: int, T: np.ndarray, R: np.ndarray, gamma=1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one iteration of the value iteration algorithm
    :param v: Current value function as an array, where (value = v[state].
    :param num_states: Number of states in the environment.
    :param T: Transition probabilities, as a 2D array.
    :param R: Rewards for certain state transitions, as a 2D array.
    :param gamma: Discount factor. 
    :return: Updated pi and v.
    """
    delta = 0
    v_new = np.copy(v)
    for s in range(num_states):
        v = np.copy(v_new)
        v_new[s] = np.max(np.sum(T[s, :, :] * (R[s, :, :] + gamma * v[np.newaxis, :]), axis=1))
        delta = max([delta, np.abs(v[s] - v_new[s])])

    v = np.copy(v_new)

    # Assuming a deterministic policy, so pi in Z^num_states instead of pi in [0, 1]^(num_states * num_actions)
    pi = np.zeros(num_states, dtype=np.int32)
    for s in range(num_states):
        pi[s] = np.argmax(np.sum(T[s, :, :] * (R[s, :, :] + gamma * v[np.newaxis, :]), axis=1))

    return pi, v
