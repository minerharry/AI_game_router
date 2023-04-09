

import time
from typing import Any, Generic, Iterable, Iterator, TypeVar
import ray
from ray.actor import ActorHandle
from baseGame import EvalGame
from game_runner_neat import GameRunner
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import SegmentState
from ray_event import RayEvent
from runnerConfiguration import RunnerConfig
from training_data import TrainingDataManager

#all this is so stupid
#ReplayRenderer can only be directly instantiated by create_replay_renderer from the main node until ray issue #32848 is resolved. Please use create_replay_renderer instead.
def create_replay_renderer(game:EvalGame,checkpoint_run_name:str,runConfig:RunnerConfig,stop_event:RayEvent|None=None)->ActorHandle:
    renderer:ActorHandle = _ReplayRenderer.remote(game,checkpoint_run_name,runConfig,stop_event=stop_event);
    player = _ReplayRenderPlayer.remote(renderer,game,runConfig,checkpoint_run_name);
    renderer.set_player.remote(player);
    return renderer


def ReplayRenderer(game:EvalGame,checkpoint_run_name:str,runConfig:RunnerConfig,stop_event:RayEvent|None=None)->ActorHandle:
    return create_replay_renderer(game,checkpoint_run_name,runConfig,stop_event=stop_event);

ReplayRenderer.remote = create_replay_renderer;








@ray.remote(num_cpus=0.01)
class _ReplayRenderer:
    def __init__(self,game:EvalGame,checkpoint_run_name:str,runConfig:RunnerConfig,stop_event:RayEvent|None=None):
        
        #generation, genome, TD
        self.replay_queue:list[int] = [];
        self.replay_registry:dict[int,Iterator[tuple[float,int,int,Any]]] = {};
        self.replay_index = -1;
        self.stop_event = stop_event;
        # self.checkpoint_name = checkpoint_run_name;
        # self.runConfig = runConfig;
        # self.game = game;
    
    def set_player(self,player:ActorHandle):
        self.player = player;
    
    def start_replay_loop(self):
        self.player.replay_loop.remote(self.stop_event);
    
        
    def register_replay(self,replay:Iterator[tuple[float,int,int,Any]]|Iterable[tuple[float,int,int,Any]]):

        if not isinstance(replay,Iterator):
            replay = iter(replay);

        key = id(replay);
        
        self.replay_queue.append(key);
        self.replay_registry[key] = replay;
        print("replay registered:",replay,"key:",key);
        return key;

    def deregister_replay(self,key:int):
        if key in self.replay_queue:
            self.replay_queue.remove(key);
            print("replay deregistered:",self.replay_registry[key],"key:",key)
            del self.replay_registry[key];


    def clear_replays(self):
        self.replay_queue = [];
        self.replay_index = -1;

    def _get_next_state(self):
        if len(self.replay_queue) == 0:
            self.replay_index = -1;
            return None;
        self.replay_index = (self.replay_index + 1) % len(self.replay_queue);
        try:
            return next(self.replay_registry[self.replay_queue[self.replay_index]]);
        except StopIteration:
            self.replay_queue.pop(self.replay_index);
            return self._get_next_state();
        


@ray.remote(num_cpus=0.1)
class _ReplayRenderPlayer():
    def __init__(self,queue,game:EvalGame,runConfig:RunnerConfig,run_name:str):
        self.queue = queue;

        self.runner = GameRunner(game,runConfig);
        self.run_name = run_name;
        assert runConfig.training_data is not None;
        self.manager:TrainingDataManager = runConfig.training_data;

    def replay_loop(self,stop_event:RayEvent|None=None):
        while stop_event is None or not stop_event.is_set():
            fitness:float; gid:int; generation:int;
            next_state = ray.get(self.queue._get_next_state.remote());
            if next_state is None:

                time.sleep(60)
                continue;
            n = next_state; 
            fitness,gid,generation,state = n
            self.manager.set_data([state]);
            print("rendering genome gid",gid,"generation",generation,"run name",self.run_name);
            try:
                self.runner.render_genome_by_id(gid,generation,self.run_name)
            except Exception as e:
                import pdb
                pdb.set_trace();
                self.runner.render_genome_by_id(gid,generation,self.run_name)