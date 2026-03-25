"""Helper functions for mathematical operations in CommPy."""


def is_prime(n: int) -> bool:
    """Check if n is a prime number."""
    if n <= 1:
        return False
    return all(n % i != 0 for i in range(2, int(n**0.5) + 1))


def modinv(a: int, p: int) -> int:
    """Extended Euclidean Algorithm for modular inverse."""
    t, newt = 0, 1
    r, newr = p, a
    while newr != 0:
        quotient = r // newr
        t, newt = newt, t - quotient * newt
        r, newr = newr, r - quotient * newr
    if r > 1:
        e = f'{a} is not invertible mod {p}'
        raise ValueError(e)
    if t < 0:
        t += p
    return t
