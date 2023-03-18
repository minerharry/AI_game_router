import time
import ray

@ray.remote
class Actor1:
    def __init__(self,data:int):
        self.data = data;

    def set_two(self,other):
        self.two = other
        print(type(self.two)) # <class 'ray.actor.ActorHandle'>
        self.two.ping.remote(); #prints "Actor2 - I've been pinged!" from Actor2's processs

    def ping(self):
        print("Actor1 - I've been pinged!")

    def get_self_data(self):
        return self.data;

    def get_other_data(self):
        return ray.get(self.two.get_data.remote());

    def increment_other(self):
        ray.get(self.two.increment_data.remote())

    def set_data(self,data):
        self.data = data;


@ray.remote
class Actor2:
    def __init__(self,one,data):
        self.one = one;
        self.data = data;
        print(type(one)); # <class 'ray.actor.ActorHandle'>
        self.one.ping.remote(); #prints "Actor1- I've been pinged!" from Actor1's processs

    def ping(self):
        print("Actor2 - I've been pinged!")

    def get_data(self):
        return self.data;

    def increment_data(self):
        self.data += 1;
        self.one.set_data.remote(self.data);



if __name__ == "__main__":
    a1 = Actor1.remote(2);
    a2 = Actor2.remote(a1,2);
    ray.get(a1.set_two.remote(a2));
    time.sleep(5);
    print(ray.get(a1.get_self_data.remote())); #prints '2'
    print(ray.get(a1.get_other_data.remote())); #prints '3'
    ray.get(a1.increment_other.remote())
    time.sleep(5) #make sure increment is complete
    print(ray.get(a1.get_self_data.remote())); #prints '3' - a2 able to increment a1's data
    print(ray.get(a1.get_other_data.remote())); #prints '4' - a1 able to increment a2's data
