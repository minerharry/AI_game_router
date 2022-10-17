from __future__ import annotations
import os
from typing import Any
import ray
from ray.actor import ActorHandle

from baseGame import EvalGame, RunGame
from runnerConfiguration import RunnerConfig

ray.init();

original = EvalGame(RunGame);

ref = ray.put(original);

@ray.remote
class IdQueue():
    def __init__(self):
        self.ids = dict[Any,int]();
        self.max_id=-1;

    def get_id(self,key):
        if key not in self.ids:
            self.max_id += 1;
            self.ids[key] = self.max_id;
            return self.max_id;        
        return self.ids[key];


@ray.remote
def task_func(inc:int,queue,*args,**kwargs):
    id = ray.get(queue.get_id.remote(os.getpid()));
    print(id);
    

a = IdQueue.remote();

# normal = Normal(17);

tasks = [task_func.remote(1,a) for i in range(15)];
print(ray.get(tasks));

