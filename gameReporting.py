
from abc import abstractmethod
import abc
import multiprocessing
from pkgutil import get_data
from typing import TypeVar,Generic

# from baseGame import RunGame


class GameReporter(abc.ABC):
    @abstractmethod
    def on_training_data_load(self,game,id): pass;
    
    @abstractmethod
    def on_start(self,game): pass;

    @abstractmethod
    def on_tick(self,game,inputs): pass;

    def on_render_tick(self,game,inputs): self.on_tick(game,inputs);

    @abstractmethod
    def on_finish(self,game): pass;


T = TypeVar('T');
class ThreadedGameReporter(GameReporter,Generic[T]): #class with nice builtin multithreading functionality
    def __init__(self):
        m = multiprocessing.Manager();
        self.data = m.Queue();

    def put_data(self,data:T):
        self.data.put(data);

    def get_data(self)->T:
        return self.data.get();

    def get_all_data(self):
        while not self.data.empty():
            yield self.get_data();