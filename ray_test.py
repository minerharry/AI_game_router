import os
import ray
import time

ray.init(ignore_reinit_error=True)

@ray.remote(max_restarts=0)
class Actor:
    def __init__(self):
        self.counter = 0

    def increment_and_possibly_fail(self):
        self.counter += 1
        time.sleep(0.2)
        if self.counter == 10:
            os._exit(0)
        return self.counter

actor = Actor.remote()

# The actor will be restarted up to 5 times. After that, methods will
# always raise a `RayActorError` exception. The actor is restarted by
# rerunning its constructor. Methods that were sent or executing when the
# actor died will also raise a `RayActorError` exception.
for _ in range(100):
    try:
        counter = ray.get(actor.increment_and_possibly_fail.remote())
        print(counter)
    except ray.exceptions.RayActorError:
        print('FAILURE')