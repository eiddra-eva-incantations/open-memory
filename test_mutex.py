import subprocess
import time

def run_worker(name, delay):
    time.sleep(delay)
    return subprocess.Popen([
        ".venv/bin/python", "-c", 
        "import sys; sys.path.append('.'); from mcp_server import encode_flagship; print(f'Worker starting'); encode_flagship('Testing concurrent 1'); encode_flagship('Testing concurrent 2'); print(f'Worker done')"
    ])

print("Starting Worker A...")
p1 = run_worker("A", 0)
print("Starting Worker B...")
p2 = run_worker("B", 1)

p1.wait()
p2.wait()
print("Both workers finished.")
