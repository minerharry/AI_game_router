
from abc import abstractmethod
import abc
import multiprocessing
from pkgutil import get_data
from typing import TypeVar,Generic

from baseGame import RunGame


class GameReporter(abc.ABC):
    
    @abstractmethod
    def on_start(self,game:RunGame): pass;

    @abstractmethod
    def on_tick(self,game:RunGame,inputs): pass;

    def on_render_tick(self,game:RunGame,inputs): self.on_tick(game,inputs);

    @abstractmethod
    def on_finish(self,game:RunGame): pass;


T = TypeVar('T');
class ThreadedGameReporter(GameReporter,Generic[T]): #class with nice builtin multithreading functionality
    def __init__(self):
        self.data = multiprocessing.Queue[T]();

    def put_data(self,data):
        self.data.put(data);

    def get_data(self):
        return self.data.get();

    def get_all_data(self):
        while self.data.not_empty:
            yield self.get_data();