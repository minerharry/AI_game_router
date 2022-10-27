# script.py
import sys
print(sys.version)
print(sys.version_info);
import ray
import time
import os

@ray.remote
def hello_world():
    return f"hello world from process {os.getpid()}"


ray.init()
refs = [hello_world.remote() for _ in range(20)];


print(ray.get(refs));