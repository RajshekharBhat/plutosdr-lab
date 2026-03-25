"""General mathematical formulas for communication theory."""
from math import log2


def shannon_entropy(probabilities: list[float]) -> float:
    """Calculate the Shannon entropy of a probability distribution.

    :param probabilities: A list of probabilities.
    :return: The Shannon entropy.
    """
    return -sum(p * log2(p) for p in probabilities if p > 0)
