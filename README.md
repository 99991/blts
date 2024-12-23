# Bruteforce LLM Test Solver

[This script](https://github.com/99991/blts/blob/main/main.py) repeatedly asks a large language model (LLM) to implement [a function with a given signature](https://github.com/99991/blts/blob/main/signature.py) until [the given tests](https://github.com/99991/blts/blob/main/tests.py) pass.
For many problems, implementing tests is less work than implementing a solution, so we can outsource the laborious task of finding a solution to an LLM.
Just write your desired function signature in `signature.py`, your tests in `test.py`, start `llama-server` and `main.py` and then go get a coffee. If you are lucky, the problem will be solved when you come back.

Currently, only a simple bruteforce approach is implemented. If time allows, I might try implementing something smarter like e.g. Language Agent Tree Search in the distant future.

# Matrix inversion example

Given the following [`signature.py`](https://github.com/99991/blts/blob/main/signature.py) and [`tests.py`](https://github.com/99991/blts/blob/main/tests.py) file, the language model will (eventually) implement a function to invert the given matrix.

#### [`signature.py`](https://github.com/99991/blts/blob/main/signature.py)

```python
def invert(A: list[list[float]]) -> list[list[float]]:
    # Invert the matrix A.
```

#### [`tests.py`](https://github.com/99991/blts/blob/main/tests.py)

```python
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
```

# How to run

1. `git clone git@github.com:99991/blts.git`
2. `cd blts`
3. Install Docker.
4. `docker build -t testimage .` to build the Docker image from the Dockerfile.
5. Install [llama.cpp](https://github.com/ggerganov/llama.cpp).
6. Download a language model. For testing, [Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/blob/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf) is a probably good enough and should run even if your computer is bad.
7. Start `llama-server` using a command similar to the following:

```bash
llama-server \
    --model qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --host 127.0.0.1 \
    --port 8080 \
    --flash-attn \
    -ngl 999 \
    --ctx-size 16384 \
    --cache-type-k q8_0 \
    --cache-type-v q8_0 \
    --parallel 10
```

8. Run [`main.py`](https://github.com/99991/blts/blob/main/main.py)
