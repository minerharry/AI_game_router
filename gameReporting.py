from __future__ import annotations
from abc import abstractmethod
import abc
import multiprocessing
from pkgutil import get_data
from typing import TypeVar,Generic,TYPE_CHECKING
if TYPE_CHECKING:
    from baseGame import RunGame

from ray.util.queue import Queue

# from baseGame import RunGame


class GameReporter():
    def on_training_data_load(self,game:RunGame,id): pass;
    
    def on_start(self,game:RunGame): pass;

    def on_tick(self,game:RunGame,inputs): pass;

    def on_render_tick(self,game:RunGame,inputs): self.on_tick(game,inputs);

    def on_finish(self,game:RunGame): pass;

    #custom reporter function; game can send signal to reporters and only the ones who have the function will receive it
    def on_signal(self,game:RunGame,signal:str,*args,**kwargs):
        if (hasattr(self,signal)):
            getattr(self,signal)(game,*args,**kwargs);


T = TypeVar('T');
class ThreadedGameReporter(GameReporter,Generic[T]): #class with nice builtin multithreading functionality
    def __init__(self,queue_type="multiprocessing"):
        if queue_type == "ray":
            self.data = Queue();
        else:
            m = multiprocessing.Manager();
            self.data = m.Queue();

    def put_data(self,data:T):
        self.data.put(data);

    def get_data(self)->T:
        return self.data.get();

    def get_all_data(self):
        while not self.data.empty():
            yield self.get_data();