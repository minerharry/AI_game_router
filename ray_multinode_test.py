# script.py
import ray
import time

@ray.remote(scheduling_strategy="SPREAD")
def hello_world():
    return f"hello world from process {os.getpid()}"


ray.init()
refs = [hello_world.remote() for _ in range(20)];

print(ray.get(refs));