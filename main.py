import shutil
import secrets
import requests
import subprocess
from pathlib import Path
from multiprocessing.pool import ThreadPool

def solve_problem(args):
    # TODO This whole thing might work better if the model tried again a few times
    # with the error message from the previous run if it failed.
    seed, signature, tests = args

    url = "http://127.0.0.1:8080/completion"

    prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Complete the following Python function:

{signature}<|im_end|>
<|im_start|>assistant
Certainly! Below is the completed function in Python:

```python
{signature}"""

    # API: https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md
    data = {
        "model": "qwen",
        "prompt": prompt,
        "temperature": 0.8,
        "seed": seed,
        "n_predict": 512,
        "stop": ["```", "```\n", "# Example"],
    }

    with requests.post(url, json=data) as r:
        r.raise_for_status()
        solution = r.json()["content"]

    code = signature + solution + tests

    returncode, stdout, stderr = run_sandboxed(code)

    result = {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "seed": seed,
        "prompt": prompt,
        "code": code,
        "solution": solution,
    }

    return result

def run_sandboxed(
    code,
    image_name="testimage",
    python_binary_path="/home/testuser/testenv/bin/python3",
    max_runtime=5.0,
):
    # TODO Might be a good idea to run multiple tests in the same
    # docker container to reduce launch overhead.

    # Make clean test directory with test code
    token = secrets.token_hex()
    container_name = "testcontainer_" + token
    directory = Path("testdirectory_" + token)
    assert not directory.exists()
    directory.mkdir(exist_ok=True)
    (directory / "test.py").write_text(code)

    # Somewhat sandboxed Docker command to run code
    command = [
        "docker", "run",
        "--rm",
        "--security-opt", "no-new-privileges",
        "--cap-drop=ALL",
        "-v", f"{directory.resolve()}:/data",
        "-w", "/data",
        "--name", container_name,
        "--memory", "1024m",
        "--memory-swap", "1024m",
        "--cpus=1",
        "--network", "none",
        image_name,
        python_binary_path,
        "test.py",
    ]

    # Start process
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)

    # Wait until process finishes or timed out
    try:
        stdout, stderr = process.communicate(timeout=max_runtime)
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        result = subprocess.run(["docker", "container", "kill", container_name])
        assert result.returncode == 0
        print("Timeout expired")

        returncode = 1
        stdout = b""
        stderr = b""
    finally:
        # Cleanup
        # TODO sometimes does not remove test directory
        shutil.rmtree(directory)

    return returncode, stdout, stderr

def main():
    with open("signature.py", encoding="utf-8") as f:
        signature = f.read()

    with open("tests.py", encoding="utf-8") as f:
        tests = f.read()

    def generate_args():
        # NB: pool.imap_unordered is not really lazy and will trigger OOM if range huge
        for seed in range(100_000):
            yield seed, signature, tests

    # Attempt to solve with four threads in parallel
    with ThreadPool(4) as pool:
        for result in pool.imap_unordered(solve_problem, generate_args()):
            # Print results
            returncode = result["returncode"]
            seed = result["seed"]
            stdout = result["stdout"]
            stderr = result["stderr"]
            code = result["code"]

            print("\033[96m")
            print("=" * 80)
            print(code)
            print("=" * 80)
            print("\033[0m")
            print(f"seed: {seed}\n")
            if stdout:
                print("stdout:")
                print(stderr.decode("utf-8"))
            if stderr:
                print("stdout:")
                print(stderr.decode("utf-8"))

            # Stop if tests passed
            if returncode == 0:
                print("\033[92mSuccess!\033[0m")
                break


if __name__ == "__main__":
    main()
