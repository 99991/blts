import random

# Make tests reproducible
random.seed(0)

# Check that NumPy is not being used
with open(__file__, encoding="utf-8") as f:
    solution = f.read().rsplit("import random\n", 1)[0]
assert "np." not in solution

def matmul(A, B):
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

for n in range(1, 10):
    for _ in range(10):
        # Create random matrix
        A = [[random.random() for _ in range(n)] for _ in range(n)]

        A_inv = invert(A)

        # A * A_inv should be the identity matrix with 1 on its diagonal and 0 otherwise
        I = matmul(A, A_inv)

        for i in range(n):
            for j in range(n):
                assert abs(I[i][j] - float(i == j)) < 1e-5
