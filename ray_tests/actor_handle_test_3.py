import time
import ray


@ray.remote
class Actor1:
    def __init__(self,data:int):
        self.data = data;
        self.two = Actor2.remote(self.data+1)
        print(type(self.two)) # <class 'ray.actor.ActorHandle'>
        self.two.set_one.remote(self);
        self.two.ping.remote(); #prints "Actor2 - I've been pinged!" from Actor2's processs

    def ping(self):
        print("Actor1 - I've been pinged!")

    def get_self_data(self):
        return self.data;

    def get_other_data(self):
        return ray.get(self.two.get_data.remote());

    def increment_other(self):
        ray.get(self.two.increment_data.remote());

    def set_data(self,data):
        self.data = data;


@ray.remote
class Actor2:
    def __init__(self,data):
        self.data = data;

    def set_one(self,one):
        self.one = one;
        print(type(one)); # <class 'actor_handle_test.Actor1>
        try:
            self.one.ping.remote(); #raises AttributeError: 'function' object has no attribute 'remote
        except:
            self.one.ping(); #prints "Actor1 - I've been pinged!" from Actor2's processs

    def ping(self):
        print("Actor2 - I've been pinged!")

    def get_data(self):
        return self.data;

    def increment_data(self):
        self.data += 1;
        try:
            self.one.set_data.remote(self.data - 1); #raises AttributeError: 'function' object has no attribute 'remote
        except:
            self.one.set_data(self.data - 1);



if __name__ == "__main__":
    a1 = Actor1.remote(2);
    time.sleep(5);
    print(ray.get(a1.get_self_data.remote())); #prints '2'
    print(ray.get(a1.get_other_data.remote())); #prints '3'
    a1.increment_other.remote()
    time.sleep(5) #make sure increment is complete
    print(ray.get(a1.get_self_data.remote())); #prints '2' - a2 unable to increment a1's data
    print(ray.get(a1.get_other_data.remote())); #prints '4' - a1 able to increment a2's data
