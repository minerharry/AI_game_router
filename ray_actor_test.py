import ray

def both_ray(cls,**kwargs):
    rem = ray.remote(cls);
    rem



@both_ray
class Ractor:
    def __init__(self):
        print("initialized");

    def ping(self):
        print("pong!");


r = Ractor();

r.ping();