import numpy as np

from CommPy import is_prime, modinv


class PrimeField:
    """Class for prime fields GF(p) where p is a prime number.
    """

    def __init__(self, p):
        if not isinstance(p, int) or p <= 1 or not is_prime(p):
            raise ValueError('p must be a prime number greater than 1.')
        self.p = p
        self.elements = np.arange(p)

    def add(self, a, b):
        """Add two elements in the field."""
        return (a + b) % self.p

    def subtract(self, a, b):
        """Subtract two elements in the field."""
        return (a - b) % self.p

    def multiply(self, a, b):
        """Multiply two elements in the field."""
        return (a * b) % self.p

    def divide(self, a, b):
        """Divide two elements in the field."""
        if b == 0:
            raise ZeroDivisionError('Division by zero is not allowed in the field.')
        # Find the modular inverse of b
        b_inv = modinv(b, self.p)
        return (a * b_inv) % self.p

    def primitive_roots(self):
        """Return all primitive roots of the field."""
        roots = []
        for g in self.elements:
            if g == 0 or g == 1:
                continue
            order = 1
            current = g
            while current != 1:
                current = (current * g) % self.p
                order += 1
            if order == self.p - 1:
                roots.append(g)
        return roots


if __name__ == '__main__':
    p = 7
    field = PrimeField(p)

    print('Field elements:', field.elements)
    print('Addition (3 + 5):', field.add(np.array([3,3]), np.array([5,5])))
    print('Subtraction (5 - 3):', field.subtract(5, 3))
    print('Multiplication (3 * 5):', field.multiply(3, 5))
    print('Division (6 / 3):', field.divide(6, 3))
    print('All primitive roots:', field.primitive_roots())
