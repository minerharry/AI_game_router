import ray
from threading import Event

@ray.remote
class _Event(Event):
    pass;

class RayEvent(Event):
    def __init__(self,*args,**kwargs):
        self.event = _Event.remote(*args,**kwargs);

    def is_set(self) -> bool:
        return ray.get(self.event.is_set.remote());

    def clear(self) -> None:
        return ray.get(self.event.clear.remote());

    def isSet(self) -> bool:
        return self.is_set();
    
    def mro(self) -> list[type]:
        return ray.get(self.event.mro.remote());

    def set(self) -> None:
        return ray.get(self.event.set.remote());

    def wait(self, timeout: float | None = ...) -> bool:
        return ray.get(self.event.wait.remote(timeout));



