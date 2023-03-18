from __future__ import annotations
import argparse
import copy
from functools import lru_cache, partial
import itertools
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import time
from typing import Any, Callable, DefaultDict, Generic, Iterable, Iterator, Literal, NamedTuple, Sequence, TypeVar
import typing

from ray import ObjectRef
from baseGame import EvalGame
from fitnessReporter import FitnessCheckpoint
from gameReporting import ThreadedGameReporter
from game_runner_neat import GameRunner

from games.smb1Py.py_mario_bros.PythonSuperMario_master.smb_game import SMB1Game
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segmentGenerator import GenerationOptions, SegmentGenerator
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c

from id_data import IdData
from interrupt import DelayedKeyboardInterrupt

from level_renderer import LevelRenderer, LevelRendererReporter

from neat.reporting import BaseReporter

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split
from torchvision.transforms.functional import hflip

import numpy as np
from tqdm import tqdm
from replay_renderer import ReplayRenderer


import ray
from ray_event import RayEvent
    
from runnerConfiguration import IOData, RunnerConfig
from search import DStarSearcher, LevelSearcher
from smb1Py_runner import NAME, generate_data, getFitness, getRunning, task_obstruction_score
from training_data import GeneratorTDSource, IteratorTDSource, SourcedShelvedTDManager,TDSource, TrainingDataManager

def log(*args,**kwargs):
    print(*args,**kwargs);

floatPos = tuple[float,float];
gridPos = tuple[int,int];

def f_x_grad(x,y,p1,p2):
  d1 = (x - p1[0])/max(0.1,dist(x,y,p1));
  d2 = (x - p2[0])/max(0.1,dist(x,y,p2));
  return d1-d2;

def f_y_grad(x,y,p1,p2):
  d1 = (y-p1[1])/max(0.1,dist(x,y,p1));
  d2 = (y-p2[1])/max(0.1,dist(x,y,p2));
  return d1-d2;

def f_smear(x,y,p1,p2):
  return 1/(dist(x,y,p1)+dist(x,y,p2));

def dist(x,y,p):
  return math.sqrt((x-p[0])**2 + (y-p[1])**2)

def circle(R):
    X = int(R)
    for x in range(-X,X+1):
        Y = int((R*R-x*x)**0.5) # outer bound for y given x
        for y in range(-Y,Y+1):
            yield(x,y);

def annulus(r,R):
    X = int(R)
    inner = list(circle(r));
    for x in range(-X,X+1):
        Y = int((R*R-x*x)**0.5) # outer bound for y given x
        for y in range(-Y,Y+1):
            if (x,y) not in inner:
                yield(x,y)

def p2gi(search_data_resolution:int,grids_bounds:tuple[int,int,int,int],grid_size:gridPos,pos:floatPos)->gridPos:
    pos = pos[0]-c.TILE_SIZE/(2*search_data_resolution),pos[1]-c.TILE_SIZE/(2*search_data_resolution);
    position_parameter = ((pos[0]-grids_bounds[0])/(grids_bounds[1]-grids_bounds[0]),(pos[1]-grids_bounds[2])/(grids_bounds[3]-grids_bounds[2]));
    closest_pixel = (round(position_parameter[0]*grid_size[0]),round(position_parameter[1]*grid_size[1]))
    return closest_pixel;

def gi2p(search_data_resolution:int,grids_bounds:tuple[int,int,int,int],grid_size:gridPos,index:gridPos)->floatPos:
    return (c.TILE_SIZE/(2*search_data_resolution)+index[0]/grid_size[0]*(grids_bounds[1]-grids_bounds[0])+grids_bounds[0],c.TILE_SIZE/(2*search_data_resolution)+index[1]/grid_size[1]*(grids_bounds[3]-grids_bounds[2])+grids_bounds[2]);

class LevelCheckpoint:
    def __init__(self,
                edge_costs:dict[tuple[gridPos,gridPos],float],
                edge_predictions:dict[tuple[gridPos,gridPos],float],
                d_checkpoint:dict,
                a_checkpoint:dict,
                previous_replay:dict[tuple[gridPos,...],list[tuple[float,int,int]]]|None=None):
        self.costs = edge_costs;
        self.pred = edge_predictions;
        self.d = d_checkpoint;
        self.a = a_checkpoint;
        self.replay = dict(previous_replay) if previous_replay else previous_replay


# from https://docs.python.org/3/library/itertools.html#itertools-recipes
# def roundrobin(*iterables):
#     "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
#     # Recipe credited to George Sakkis
#     num_active = len(iterables)
#     nexts = itertools.cycle(iter(it).__next__ for it in iterables)
#     while num_active:
#         try:
#             for next in nexts:
#                 yield next()
#         except StopIteration:
#             # Remove the iterator we just exhausted from the cycle.
#             num_active -= 1
#             nexts = itertools.cycle(itertools.islice(nexts, num_active))

# def cycle(iterable):
#     # cycle('ABCD') --> A B C D A B C D A B C D ...
#     saved = []
#     for element in iterable:
#         yield element
#         saved.append(element)
#     while saved:
#         for element in saved:
#               yield element

def roundrobin(*iterables:Iterable):
    return iter_roundrobin(*iterables);

class iter_roundrobin(Iterator):
    def __init__(self,*iterables:Iterable):
        self.iterables = iterables;
        self.num_active = len(iterables);
        self.index = -1;
        self.iterators = [iter(iterable) for iterable in iterables]        
    
    def __next__(self):
        if self.num_active == 0:
            raise StopIteration;
        self.index = (self.index + 1) % self.num_active;
        try:
            return next(self.iterators[self.index]);
        except StopIteration:
            self.num_active -= 1;
            self.iterators.pop(self.index);
            return next(self);


class LevelPlayer(BaseReporter):
    def set_level_info(self,view_distance:int,tile_scale:int):
        self.view_distance = view_distance;
        self.tile_scale = tile_scale;
    # def set_NEAT_player(self,game:EvalGame,runConfig:RunnerConfig,run_name:str,
    #         player_view_distance:None|int=None,
    #         player_tile_scale:None|int=None,
    #         config_override=None,
    #         checkpoint_run_name:str|None=None,
    #         extra_training_data:Iterable[SegmentState]|None=None,
    #         extra_training_data_gen:Callable[[],Iterable[SegmentState]]|None=None,
    #         fitness_save_path:str|Path|None=None):

    #     self.runConfig = runConfig;
    #     self.neat_config_override = config_override;
    #     self.run_name = run_name;
    #     self.checkpoint_run_name = checkpoint_run_name if checkpoint_run_name else self.run_name;

    #     self.tdat = SourcedShelvedTDManager[SegmentState]('smb1Py',run_name);
    #     self.runConfig.training_data = self.tdat;

    #     self.extra_dat_gen = extra_training_data_gen if extra_training_data_gen else ((lambda: extra_training_data) if extra_training_data else None);
    #     self.game = game;
    #     self.runConfig.generations = 1;
    #     self.gamerunner = GameRunner(self.game,self.runConfig);

    #     if fitness_save_path:
    #         self.task_reporter = TaskFitnessReporter(fitness_save_path,queue_type=self.runConfig.queue_type);
    #     else:
    #         self.task_reporter = TaskFitnessReporter(queue_type=self.runConfig.queue_type);
    #     self.game.register_reporter(self.task_reporter);

    #     self.view_distance = player_view_distance if player_view_distance else self.runConfig.view_distance;
    #     self.tile_scale = player_tile_scale if player_tile_scale else self.runConfig.tile_scale;

    def start_generation(self, generation):
        self.generation = generation;

    def end_generation(self, config, population, species_set):
        self.reporter_data = self.task_reporter.get_all_conserved_data();



    def set_fixed_net(self,model:torch.nn.Module,used_grids:str|list[str],endpoint_padding:gridPos,minimum_viable_distance:float,maximum_viable_distance:float):
        '''initialize fixed net and use parameters.

            used grids: what grids from the map data (collision, bricks, enemies, etc) the model uses
            endpoint padding: how much space in the image (in pixel coordinates, dependent on resolution) is required around player start and task positions
            minimum viable distance: how far apart player start and task must be for reliable data
        '''
        if not(isinstance(used_grids,list)):
            used_grids = [used_grids];
        self.fixed_grids = used_grids;
        self.fixed_net = model;
        self.min_task_dist = minimum_viable_distance;
        self.max_task_dist = maximum_viable_distance;
        self.fixed_padding = endpoint_padding;

    def eval_fixed_net(self,in_grids,start:gridPos,task:gridPos,size:gridPos)->float:
        
        left,right = (start[0],task[0]) if start[0] < task[0] else (task[0],start[0]);
        top,bottom = (start[1],task[1]) if start[1] < task[1] else (task[1],start[1]);

        #left,right,top,bottom
        bounds = [max(left-self.fixed_padding[0],0),min(right+self.fixed_padding[0],size[0]-1),max(top-self.fixed_padding[1],0),min(bottom+self.fixed_padding[1],size[1]-1)];
        x_grad = np.array([[f_x_grad(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();
        y_grad = np.array([[f_y_grad(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();
        smear = np.array([[f_smear(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();

        in_grids = np.array(in_grids)[:,bounds[0]:bounds[1]+1,bounds[2]:bounds[3]+1]
        
        grids:np.ndarray = np.concatenate((in_grids,[x_grad,y_grad,smear]),axis=0)

        grids = np.expand_dims(grids,0);

        full = Tensor(grids);
        return self.fixed_net(full).item();

    # def eval_NEAT_player(self,tasks:list[list[floatPos]],level:SegmentState):
    #     '''returns the fitness values (for ALL players) for each leg of the task'''


    #     return result;



    def get_checkpoint_data(self,saved_replay:dict[tuple[gridPos,...],list[tuple[float,int,int]]]|None=None):
        return LevelCheckpoint(self.costs,self.predictions,self.d_searcher.get_checkpoint_data(),self.a_searcher.get_checkpoint_data(),previous_replay=saved_replay);

    @classmethod
    def task_path_to_state(cls,level:SegmentState,task_path:Sequence[floatPos],id:Any=None):
        state = level.deepcopy();
        task_path = task_path[1:];
        state.task = task_path[0];
        state.task_path = task_path;
        state.source_id = id;
        return state;

    #given a level (no task, no task_bounds), a goal block or set of blocks
    #offset downscale means with how much resolution are potential task blocks generated.
    def play_level(self,
            game:EvalGame,
            level:SegmentState,
            goal:floatPos|list[floatPos],
            fitness_reporter:TaskFitnessReporter,
            training_dat_per_gen=50,
            search_data_resolution=5,
            task_offset_downscale=2,
            search_checkpoint:LevelCheckpoint|None=None,
            checkpoint_save_location="play_checkpoint.chp",
            fitness_aggregation_type="max",
            render_progress=True,
            replay_renderer:ReplayRenderer|None = None,
            multiprocessing_type:Literal["ray","multiprocessing"]="ray")->TDSource[SegmentState]:
        self.generation = -1;
        gen = self._yield_NEAT_data(game,level,goal,fitness_reporter,training_dat_per_gen,search_data_resolution,task_offset_downscale,search_checkpoint,checkpoint_save_location,fitness_aggregation_type,render_progress,replay_renderer,multiprocessing_type)
        self.source = IteratorTDSource[SegmentState](gen)
        if next(gen) == True:
            return self.source;
        else:
            raise Exception(); ##shrug

    def _yield_NEAT_data(self,
            game:EvalGame,
            level:SegmentState,
            goal:floatPos|list[floatPos],
            fitness_reporter:TaskFitnessReporter,
            training_dat_per_gen:int,
            search_data_resolution:int,
            task_offset_downscale:int,
            search_checkpoint:LevelCheckpoint|None,
            checkpoint_save_location:str|Path|os.PathLike,
            fitness_aggregation_type:str,
            render_progress:bool,
            replay_renderer:ReplayRenderer|None,
            multiprocessing_type:Literal["ray","multiprocessing"]
            ):
        self.multi = multiprocessing_type
        if self.multi not in ["multiprocessing","ray"]:
            raise TypeError(f"library type {self.multi} not known");

        render_best = replay_renderer is not None;
        replay_ref = None;
        
        self.game = game;
        self.task_reporter = fitness_reporter

        fitness_from_list = {
            "max":max,
            "median":np.median,
            "mean":np.average,
            "q3":lambda l: np.quantile(l,0.75),
            "q1":lambda l: np.quantile(l,0.25),
            "min":min
        }[fitness_aggregation_type];

        if search_data_resolution % task_offset_downscale != 0:
            log("WARNING: Attempting to downscale task resolution by a nonfactor of the data resolution. Setting task resolution to data resolution.")
            task_offset_downscale = 1;

        ### Extract level info for routing purposes
        log("--EXTRACTING LEVEL INFO--");
        log("game startup");
        data_game = Segment()
        data_game.startup(0,{c.LEVEL_NUM:1},initial_state=level);

        log("acquiring map data")
        gdat = data_game.get_game_data(self.view_distance,self.tile_scale);
        mdat = data_game.get_map_data(search_data_resolution);

        del data_game;

        self.player_start:floatPos = gdat['pos'];
        search_grids = [np.array(mdat[g]) for g in self.fixed_grids];
        grid_size = search_grids[0].shape;
        grids_bounds:tuple[int,int,int,int] =  mdat['grid_bounds'];

        log('start position  (continuous):',self.player_start,'discrete grid size:',grid_size,'discrete grids\' bounds:',grids_bounds);

        if not isinstance(goal,list):
            goal = [goal];

        def in_bounds(pos:gridPos):
            return pos[0] >= 0 and pos[0] < grid_size[0] and pos[1] >= 0 and pos[1] < grid_size[1];

        def cost_from_fitness(fitness:float):
            if (fitness <= 0):
                return float('inf');
            return 1/fitness;

        def pos_to_grid_index(pos:floatPos):
            return p2gi(search_data_resolution,grids_bounds,grid_size,pos);

        # def test(pos):
        #     pos = pos[0]-c.TILE_SIZE/(2*search_data_resolution),pos[1]-c.TILE_SIZE/(2*search_data_resolution);
        #     position_parameter = ((pos[0]-grids_bounds[0])/(grids_bounds[1]-grids_bounds[0]),(pos[1]-grids_bounds[2])/(grids_bounds[3]-grids_bounds[2]));
        #     closest_pixel = (round(position_parameter[0]*grid_size[0]),round(position_parameter[1]*grid_size[1]))
        #     return closest_pixel;

        self.pos_to_grid_index = pos_to_grid_index;

        # for _ in tqdm(range(500)):
        #     p = (random.random()*5000,random.random()*5000)
        #     assert self.pos_to_grid_index(p) == test(p);

        def grid_index_to_pos(index:gridPos):
            return gi2p(search_data_resolution,grids_bounds,grid_size,index);

        self.grid_index_to_pos = grid_index_to_pos;

        def test2(index):
            return (c.TILE_SIZE/(2*search_data_resolution)+index[0]/grid_size[0]*(grids_bounds[1]-grids_bounds[0])+grids_bounds[0],c.TILE_SIZE/(2*search_data_resolution)+index[1]/grid_size[1]*(grids_bounds[3]-grids_bounds[2])+grids_bounds[2]);

        # for _ in tqdm(range(500)):
        #     p = (int(random.random()*500),int(random.random()*500))
        #     assert self.grid_index_to_pos(p) == test2(p);
        
        if search_checkpoint is None:
            self.costs = dict[tuple[gridPos,gridPos],float]();
            self.predictions = dict[tuple[gridPos,gridPos],float]();
        else:
            self.costs = search_checkpoint.costs;
            self.predictions = search_checkpoint.pred;

        def get_cost(start:gridPos,task:gridPos):
            if mdat['collision_grid'][task[0]][task[1]] != 0:
                return float('inf');
            if (start,task) in self.costs:
                return self.costs[start,task];
            elif (start,task) not in self.predictions:
                self.predictions[start,task] = cost_from_fitness(dist(*grid_index_to_pos(start),grid_index_to_pos(task))*self.eval_fixed_net(search_grids,start,task,grid_size));
            return self.predictions[start,task];          


        ### Initialize D* Searcher (run in reverse through the level)
        log("initializing d*")
        task_offsets = [(s[0]*task_offset_downscale,s[1]*task_offset_downscale) for s in annulus(self.min_task_dist/task_offset_downscale,self.max_task_dist/task_offset_downscale)]
        log('offsets generated',len(task_offsets));

        def d_star_heuristic(p1:gridPos|Literal['goal'],p2:gridPos|Literal['goal']): #made to be bidirectional since lazy
            if isinstance(p1,str):
                try:
                    return min([d_star_heuristic(pos,p2) for pos,_ in d_star_pred(p1)]);
                except Exception as e:
                    log(p1)
                    log(list(d_star_succ(p1)));
                    raise e;
            if isinstance(p2,str):
                return d_star_heuristic(p2,p1);

            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5;

        def d_star_succ(pos:gridPos|Literal['goal']): #returns list of (succ,cost(pos,succ))
            if (isinstance(pos,str)):
                raise Exception("cannot find successors of the goal");
            # log("d_preds");
            for offset in tqdm(task_offsets,leave=False):
                task = (pos[0]+offset[0],pos[1]+offset[1]);
                if (in_bounds(task)):
                    yield (task,get_cost(pos,task));
            
            if pos in goal_idxs:
                yield ('goal',0);

        def d_star_pred(pos:gridPos|Literal['goal']): #returns list of (pred,cost(pred,pos))
            # log("d_succs");
            if pos == 'goal':
                for i in goal_idxs:
                    yield i,0;
                return;

            for offset in tqdm(task_offsets,leave=False):
                task = (pos[0]+offset[0],pos[1]+offset[1]);
                if (in_bounds(task)):
                    yield (task,get_cost(task,pos));

        start_idx = pos_to_grid_index(self.player_start);
        goal_idxs = [pos_to_grid_index(g) for g in goal];

        log('routing from start',start_idx,'to goals',goal_idxs,'in level of gridded size',grid_size);

        check = search_checkpoint.d if search_checkpoint is not None else None;
        self.d_searcher = DStarSearcher[gridPos|Literal['goal']](d_star_heuristic,start=start_idx,goal='goal',pred=d_star_pred,succ=d_star_succ,checkpoint=check);

        ### Initialize A* Searcher (runs the level forward, selects training data)
        log("initializing a*")

        def a_star_heuristic(path:tuple[gridPos,...]):
            self.d_searcher.explore_from_position(path[-1]);
            return self.d_searcher.rhs[path[-1]];

        def a_star_succ(path:tuple[gridPos,]):
            end = path[-1];
            for offset in task_offsets:
                new = (end[0]+offset[0],end[1]+offset[1]);
                if not in_bounds(new):
                    continue;
                newpath = list(path);
                newpath.append(new);
                yield tuple(newpath);

        def a_star_cost(p1:tuple[gridPos,...],p2:tuple[gridPos,...]):
            return get_cost(p1[-1],p2[-1]);

        check = search_checkpoint.a if search_checkpoint is not None else None;
        self.a_searcher = LevelSearcher[tuple[gridPos,...],gridPos]((start_idx,),lambda x: x[-1] in goal_idxs,a_star_heuristic,lambda node: node[-1],a_star_succ,a_star_cost,frustration_multiplier=0.02,checkpoint=check);


        self.renderer = None;
        if render_progress:
            log("activating renderer")
            self.renderer = LevelRenderer(level,point_size=3,path_width=2,active_path_width=3);
            self.renderReporter = LevelRendererReporter(self.source,queue_type=self.multi);
            self.game.register_reporter(self.renderReporter);
            

        yield True; #ready signal

        ### ROUTE + PLAY LEVEL
        log("beginning routing")

        log("starting d search")

        self.d_searcher.step_search();

        level_finished = False;
        winning_path = None

        # log(self.a_searcher.completed_edges);
        # log(self.d_searcher);
        # log(self.a_searcher);

        current_replay:dict[tuple[gridPos,...],list[tuple[float,int,int]]]|None = getattr(search_checkpoint,"replay",None);

        while not level_finished:
            log("checkpoint saved")
            ##save checkpoint
            if render_best:
                save_data = self.get_checkpoint_data(saved_replay=current_replay);
            else:
                save_data = self.get_checkpoint_data();
            try:
                pickle.dumps(save_data);
                with open(checkpoint_save_location,'wb') as f, DelayedKeyboardInterrupt():
                    pickle.dump(save_data,f);
            except:
                raise Exception("Error while trying to pickle checkpoint data");
            

            if render_best and current_replay:
                log("activating replay")
                rators = [];
                for path,bests in current_replay.items():
                    print(path);
                    state = self.task_path_to_state(level,[grid_index_to_pos(p) for p in path]);
                    rators.append([b + (state,) for b in sorted(bests,reverse=True)]);
                rators.sort(key = lambda l: l[0][0]);
                log("Best genome paths:", [r[0] for r in rators])
                if replay_ref is not None:
                    replay_renderer.deregister_replay.remote(ray.get(replay_ref));
                replay_ref = replay_renderer.register_replay.remote(roundrobin(*rators))


            while True:
                time.sleep(5);

            log("retrieving best edges from A*")
            top_paths = self.a_searcher.sorted_edges()
            
            # log('top ten scores:',[(self.a_searcher.sort_key(e),e) for e in top_paths[:10]]);

            top_paths = top_paths[:training_dat_per_gen];
            log(len(top_paths),"edges retrieved");

            player_paths = [[grid_index_to_pos(task) for task in path] for _prev,path in top_paths];
            

            log("NEAT-player attempting level segments");
            

            ### Evaluating neat player, used to be its own function, copy-pasted
            tasks = player_paths

            log("evaluating NEAT-player with training data",len(tasks),'ex:',tasks[0],'and level',level);
            data:list[SegmentState] = [];
            for i,task_path in enumerate(tasks):
                state = self.task_path_to_state(level,task_path,id=i);
                data.append(state);
            

            renderProcess = None;
            if self.renderer is not None:
                reached_idxs = [p[1][-1] for p in self.a_searcher.completed_edges];
                failed_idxs = [p for _,p in self.costs.keys() if p not in reached_idxs];

                reached = [self.grid_index_to_pos(idx) for idx in reached_idxs];
                failed = [self.grid_index_to_pos(idx) for idx in failed_idxs];
                paths:dict[int,Sequence[tuple[float,float]]] = {state.source_id:[self.player_start,state.task,*state.task_path] for state in data};

                self.renderer.set_annotations(reached,failed,paths);

                self.renderReporter.reset_paths();
                if self.multi == "multiprocessing":
                    self.kill_event = multiprocessing.Event();
                    renderProcess = multiprocessing.Process(target=self.renderReporter.render_loop,args=[self.renderer,self.kill_event]);
                    renderProcess.start();
                elif self.multi == "ray":
                    self.kill_event = RayEvent();
                    # ray.util.inspect_serializability(self.renderReporter);
                    renderProcess = self.renderReporter.ray_render_loop.remote(self.renderReporter,self.renderer,self.kill_event);
                else:
                    raise Exception();



            ### Generation n

            yield data;

            ### Generation n+1



            all_d = self.reporter_data or [];
            result:list[tuple[int,list[tuple[tuple[floatPos, floatPos | Literal['complete']], float]]]] = [(d.data[2],d.data[1]) for d in all_d if d.data[0] == self.source];
            
            log("Neat player evaluated;",len(result),"data collected");

            if renderProcess is not None:
                self.kill_event.set();
                if self.multi=="multiprocessing":
                    renderProcess.join(); #type: ignore
                else:
                    ray.get(renderProcess); #type: ignore
            
            all_fitnesses = result
            log("fitnesses calculated")

            self.a_searcher.register_attempts(top_paths);


            best_segments:dict[tuple[gridPos,gridPos],list[float]] = DefaultDict(lambda:[]);
            best_paths:dict[tuple[gridPos,...],list[float]] = DefaultDict(lambda:[])
            completed_paths:list[tuple[gridPos]] = [];

            best_players:dict[tuple[gridPos,...],list[tuple[float,int,int]]] = DefaultDict(lambda:[]);
            #fitness,genome id,generation
            
            for gid,fitness_list in tqdm(all_fitnesses,desc="processing fitnesses: "):
                acc_fitness = 0;
                acc_path:list[gridPos] = [];
                for (start,end),fitness in fitness_list:
                    if end == 'complete':
                        # log("completed")
                        if tuple(acc_path) not in completed_paths:
                            completed_paths.append(tuple(acc_path)) 
                            if (pos_to_grid_index(start) in goal_idxs):
                                level_finished = True;
                                winning_path = acc_path;

                    else:
                        start = pos_to_grid_index(start);
                        end = pos_to_grid_index(end);
                        if start == end:
                            log("ERROR: start and end should not be the same; from path",fitness_list)
                            continue;

                        best_segments[start,end].append(fitness);

                        acc_fitness += fitness;

                        if len(acc_path) == 0:
                            acc_path.append(start);
                        acc_path.append(end);
                        
                        best_paths[tuple(acc_path)].append(acc_fitness);
                        if render_best: best_players[tuple(acc_path)].append((acc_fitness,gid,self.generation-1));
            log("total paths:",len(best_paths));
            log("completed paths:",len(completed_paths));

            d_updates:list[tuple[gridPos|Literal['goal'],gridPos|Literal['goal'],float,float]] = [];
            a_updates:list[tuple[tuple[gridPos,...],float]] = [];

            for (start,end),fitnesses in best_segments.items():
                fitness = fitness_from_list(fitnesses);
                old_cost = get_cost(start,end);
                self.costs[start,end] = cost_from_fitness(fitness);
                d_updates.append((start,end,old_cost,cost_from_fitness(fitness)));

            for path,fitnesses in best_paths.items():
                fitness = fitness_from_list(fitnesses);
                a_updates.append((path,cost_from_fitness(fitness)));
            
            current_replay = best_players;


            log('running d* iteration')
            self.d_searcher.update_costs(d_updates);

            log('running A* iteration')
            log(f'{len(completed_paths)} newly completed paths: {completed_paths}')
            for path in completed_paths:
                self.a_searcher.complete_edge((tuple(path[:-1]),tuple(path)));

            self.a_searcher.update_scores(a_updates);
        log("checkpoint saved")
        ##save checkpoint
        save_data = self.get_checkpoint_data();
        with open(checkpoint_save_location,'wb') as f:
            pickle.dump(save_data,f);
        if render_progress:
            self.game.deregister_reporter(self.renderReporter)

        self.winning_path = winning_path;
        log("winning path found! Path:",self.winning_path);

task_out_k = list[tuple[tuple[floatPos,floatPos|Literal['complete']],float]]#I'm so sorry
class TaskFitnessReporter(BaseReporter,ThreadedGameReporter[IdData[tuple[TDSource,task_out_k,int]]]): 
    def __init__(self,save_path=None,**kwargs):
        super().__init__(**kwargs);
        self.save_path = Path(save_path) if save_path else None;
        self.generation = None;
        #out: tuple[source, out, genome id]
        self.data_list:list[IdData[tuple[TDSource,task_out_k,int]]]|None = None
        self.data = None;
        self.data_source:TDSource|None = None;
        self.data_id = None;

    # def add_capture_source(self,*source:TDSource):
    #     self.captures.extend(source);
    
    # def remove_capture_source(self,*source:TDSource):
    #     [self.captures.remove(s) for s in source];


    #NOTE: EXECUTED IN PARALLEL PROCESSES
    def on_training_data_load(self, game: SMB1Game, id:int|None):
        if self.data_list is not None:
            self.put_all_data(*self.data_list);
        self.data_list = [];
        if id is not None and self.data_id != id:
            self.data_id = id;
            t:SourcedShelvedTDManager = game.runConfig.training_data #type: ignore
            s = t.get_datum_source(id)
            self.data_source = s;

    def on_start(self, game: SMB1Game, genome_id:int):
        self.previous_task:floatPos = game.getMappedData()['pos'];
        self.current_task:floatPos = game.getMappedData()['task_position'];
        self.current_genome = genome_id;
        self.current_fitness = game.getFitnessScore();
        self.current_data:list[tuple[tuple[floatPos,floatPos|Literal['complete']],float]] = [];

    def on_tick(self, game: SMB1Game, inputs, finish = False):
        task:floatPos = game.getMappedData()['task_position'];
        if (task != self.current_task or finish):
            out_fitness = game.getFitnessScore() - self.current_fitness;
            self.current_data.append(((tuple(self.previous_task),tuple(self.current_task)),out_fitness));
            self.previous_task = self.current_task;

            self.current_task = task;
            self.current_fitness = game.getFitnessScore();

    def on_finish(self, game: SMB1Game):

        self.on_tick(game,None,finish=True);
        if game.getMappedData()['task_path_complete']:
            self.current_data.append(((tuple(self.previous_task),'complete'),-1));
        assert self.data_list is not None
        assert self.data_source is not None
        assert self.data_id is not None
        self.data_list.append(IdData(self.data_id,(self.data_source,self.current_data,self.current_genome)));

    def start_generation(self, generation):
        log(f"Task Fitness Reporter - generation {generation} started");
        self.generation = generation;
        super().get_all_data(); #flush data contents
        self.data = None;

    def end_generation(self, config, population, species_set):
        log(f"Task Fitness Reporter - generation {self.generation} ended");
        if self.generation is not None and self.save_path:
            data = list(self.get_all_conserved_data());
            out:dict[int,dict[int,list[float]]] = DefaultDict(lambda: {});
            for d in data:
                out[d.id][d.data[2]] = ([f[1] for f in d.data[1]]);
            try:
                out_path = self.save_path/f"gen_{self.generation}";
                checkpoint = FitnessCheckpoint(out);
                checkpoint.save_checkpoint(out_path);
            except Exception as e:
                print(e);

    def get_all_data(self):
        raise NotImplementedError("unable to pull all data from fitness reporter; if you want to pull data from a source, add a capture source");

    def get_all_conserved_data(self):
        log("obtaining conserved data")
        if self.data is not None:
            log("already obtained data; returning",len(self.data),"data");
            return self.data
        if self.data_list is not None:
            self.put_all_data(*self.data_list);
            self.data_list = None;
        out:list[IdData[tuple[TDSource,task_out_k,int]]] = [];
        log("getting conserved data...");
        for d in list(super().get_all_data()):
            out.append(d);
            # self.put_data(d);
        self.data = out;
        log("conserved data obtained:",len(self.data))
        return out;


def train_loop(dataloader:DataLoader, model, loss_fn, optimizer,weight_fun:Callable[[Tensor,Tensor,Tensor],Tensor]|None=None):
    loss = None;
    for (X,y) in tqdm(dataloader,leave=False):
        # Compute prediction and loss
        pred = model(X)
        # log(X.shape);
        # log(y.shape);
        # log(pred.shape);

        # Backpropagation
        optimizer.zero_grad(set_to_none=True)
        if weight_fun:
          loss = loss_fn(pred, y, weight_fun(X,y,pred));
        else:
          loss = loss_fn(pred,y);
        loss.backward()
        optimizer.step()
        
    tqdm.write(f"Train loss: {loss:>7f}");


def test_loop(dataloader:DataLoader, model, loss_fn,weight_fun:Callable[[Tensor,Tensor,Tensor],Tensor]|None=None):
    num_batches = 0;
    test_loss = 0
    loss = 0;
    with torch.no_grad():
        for X, y in tqdm(dataloader,leave=False):
            num_batches += 1;
            # log(X.shape);
            pred = model(X)
            # log(y);
            if weight_fun:
              loss = loss_fn(pred, y, weight_fun(X,y,pred));
            else:
              loss = loss_fn(pred,y);
            test_loss += loss.item()
            # correct += np.array(pred-y).mean();
    if num_batches != 0:
        test_loss /= num_batches
        tqdm.write(f"Test loss: {test_loss:>8f} \n")

class ModelTunerReporter(BaseReporter):
    def __init__(self,
            model:torch.nn.Module,
            used_grids:str|list[str],
            endpoint_padding:gridPos,
            minimum_viable_distance:float,
            maximum_viable_distance:float,
            view_distance:int,
            tile_scale:int,
            search_data_resolution:int,
            fitness_reporter:TaskFitnessReporter,
            training_data_manager:TrainingDataManager[SegmentState],
            learning_rate:float,
            epochs_per_gen:int,
            batch_size:int,
            test_dl_fraction:float=0.1,
            device_type:str="cpu",
            fitness_aggregation_type:str="max",
            model_save_path:str|None=None,
            model_save_format:str="model_gen{0}.model",
            do_hflip=True):
        self.model = model;
        self.fitness_from_list = {
            "max":max,
            "median":np.median,
            "mean":np.average,
            "q3":lambda l: np.quantile(l,0.75),
            "q1":lambda l: np.quantile(l,0.25),
            "min":min
        }[fitness_aggregation_type];
        if not(isinstance(used_grids,list)):
            used_grids = [used_grids];
        self.fixed_grids = used_grids;
        self.min_task_dist = minimum_viable_distance;
        self.max_task_dist = maximum_viable_distance;
        self.view_distance = view_distance;
        self.tile_scale = tile_scale;
        self.search_resolution = search_data_resolution;
        self.fixed_padding = endpoint_padding;
        self.reporter = fitness_reporter;
        self.manager = training_data_manager;
        self.lr = learning_rate;
        self.epochs = epochs_per_gen;
        self.batch_size = batch_size;
        self.test_fraction = test_dl_fraction;
        self.device_type = device_type;
        self.save_path = Path(model_save_path) if model_save_path else None;
        self.save_format = model_save_format;
        self.generation = -1;
        self.do_hflip = do_hflip

    def get_search_grids(self,td_id:int):
        level = self.manager[td_id];

        data_game = Segment()
        data_game.startup(0,{c.LEVEL_NUM:1},initial_state=level);

        # log("acquiring map data")
        gdat = data_game.get_game_data(self.view_distance,self.tile_scale);
        mdat = data_game.get_map_data(self.search_resolution);

        del data_game;

        self.player_start:floatPos = gdat['pos'];
        search_grids = [np.array(mdat[g]) for g in self.fixed_grids];
        grid_size = search_grids[0].shape;
        grids_bounds:tuple[int,int,int,int] =  mdat['grid_bounds'];

        return search_grids,grid_size,grids_bounds

    def start_generation(self, generation):
        self.generation = generation;
        # return super().start_generation(generation)

    def end_generation(self, config, population, species_set):
        log("Model tuner - generation complete")
        all_data = self.reporter.get_all_conserved_data();
        log("accumulated data from this generation:",len(all_data))
        self.update_model(all_data);
        if self.save_path:
            torch.save(self.model,self.save_path/self.save_format.format(self.generation))

        
    def get_input_grids(self,start:gridPos,task:gridPos,size:gridPos,in_grids):
        left,right = (start[0],task[0]) if start[0] < task[0] else (task[0],start[0]);
        top,bottom = (start[1],task[1]) if start[1] < task[1] else (task[1],start[1]);

        #left,right,top,bottom
        bounds = [max(left-self.fixed_padding[0],0),min(right+self.fixed_padding[0],size[0]-1),max(top-self.fixed_padding[1],0),min(bottom+self.fixed_padding[1],size[1]-1)];
        x_grad = np.array([[f_x_grad(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();
        y_grad = np.array([[f_y_grad(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();
        smear = np.array([[f_smear(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();

        in_grids = np.array(in_grids)[:,bounds[0]:bounds[1]+1,bounds[2]:bounds[3]+1]
        
        grids:np.ndarray = np.concatenate((in_grids,[x_grad,y_grad,smear]),axis=0)

        # grids = np.expand_dims(grids,0); #don't expand, because training batches

        return Tensor(grids);
    

    def update_model(self,data:list[IdData[tuple[TDSource,task_out_k,int]]]):
        log("Tuning Model from generation's fitnesses");
        try:
            mapped_fitnesses:dict[int,dict[tuple[floatPos,floatPos],list[float]]] = DefaultDict(lambda: DefaultDict(lambda: []));
            for d in tqdm(data,desc="extracting data..."):
                id = d.id;
                dd = d.data[1];
                for (step,fitness) in dd:
                    if step[1] == "complete":
                        continue;
                    mapped_fitnesses[id][step].append(fitness); #type: ignore

            mapped_aggregates:dict[int,dict[tuple[floatPos,floatPos],float]] = DefaultDict(lambda:{});
            for id,steps in tqdm(mapped_fitnesses.items(),desc="aggregating fitnesses"):
                for step,fitnesses in steps.items():
                    mapped_aggregates[id][step] = self.fitness_from_list(fitnesses);
        

            shaped_grids:dict[tuple[int,...],list[tuple[Tensor,Tensor]]] = DefaultDict(lambda:[]);
            for training_id in tqdm(mapped_aggregates,desc="extracting training data info"):
                search_grids,grid_size,grids_bounds = self.get_search_grids(training_id);
                # def grid_index_to_pos(grid_index:gridPos):
                #     return gi2p(self.search_resolution,grids_bounds,grid_size,grid_index);
                def pos_to_grid_index(pos:floatPos):
                    return p2gi(self.search_resolution,grids_bounds,grid_size,pos);

                for (start,task),fitness in mapped_aggregates[training_id].items():
                    gridstart = pos_to_grid_index(start);
                    gridend = pos_to_grid_index(task);

                    base_grids = self.get_input_grids(gridstart,gridend,grid_size,search_grids);

                    d = dist(*start,task);

                    eq_grids = [base_grids];
                    if self.do_hflip:
                        eq_grids.append(hflip(base_grids))

                    for input_grids in eq_grids:
                        shaped_grids[input_grids.shape[1:]].append((input_grids.to(self.device_type),Tensor([fitness/d]).to(self.device_type)));

            # log(shaped_grids);
            
            sets = [random_split(s,[len(s)-int(len(s)*self.test_fraction),int(len(s)*self.test_fraction)], generator=torch.Generator().manual_seed(0)) for s in shaped_grids.values()];

            train_dataloader = [b for loader in [DataLoader(data[0],batch_size=self.batch_size) for data in sets] for b in loader];
            test_dataloader = [b for loader in [DataLoader(data[1],batch_size=self.batch_size) for data in sets] for b in loader ];
            # log(len(train_dataloader))
            # log(train_dataloader);
            # log(test_dataloader);


            loss_fn = torch.nn.MSELoss()
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,nesterov=True,momentum=0.8);
            for t in tqdm(range(self.epochs),desc="Model Fine-Tuning: "):
                tqdm.write(f"Epoch {t+1}\n-------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer)
                test_loop(test_dataloader, model, loss_fn)
            # import code
            # code.interact(local=locals());
        except Exception as e:
            raise e;
            # import code
            # code.interact(local=locals());
        # finally:
        #     raise Exception();
        log("Model Tuned!");

    
        

                


        
    
if __name__== "__main__":

    open("console_out.txt","w");
    def log2(*args,**kwargs):
        print(*args,**kwargs);
        print(*args,**kwargs,file=open("console_out.txt","a"))

    log = log2;
        

    
    ### PREPARE RAY / MULTIPROCESSING / CMDLINE STUFF ###
    import ray
    from ray.util.placement_group import placement_group,placement_group_table
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    parser = argparse.ArgumentParser(
                    prog = 'Play SMBPy Level',
                    description = 'attempt to play a level using multiprocessing w/ ray')
    parser.add_argument('ip');
    parser.add_argument('-L', '--local_display',action='store_true')

    args = parser.parse_args();
    ip = args.ip;
    local_display = args.local_display;

    log("Displaying locally" if local_display else "Displaying remotely");

    

    if not local_display:
        ray.init(address=ip);
        log("waiting for display node...");
        num_display = 0
        while num_display < 2:
            r = ray.cluster_resources();
            if "Display" in r:
                num_display = r["Display"]
            time.sleep(5);
        log("display node obtained, display cores available:",num_display);
        log("cluster nodes:",ray.nodes());

        basic_cores = ray.cluster_resources()["CPU"]-num_display-2; #two extra cores for whatever

        cpu_bundles = [{"CPU":1.0} for _ in range(int(basic_cores))];
        display_bundles = [{"Display":0.01,"CPU":1} for _ in range(int(num_display) - 1)];

        total_bundles = cpu_bundles + display_bundles
        group = placement_group(total_bundles,strategy="SPREAD");
        ray.get(group.ready());
        log(placement_group_table(group));
        st = PlacementGroupSchedulingStrategy(group);
    else:
        ray.init(resources={"Display":100});



    ### LOAD NEAT PLAYER ###

    game = EvalGame(SMB1Game,auto_detect_render=False,num_rendered_processes=2) if local_display else EvalGame(SMB1Game,auto_detect_render=True);

    inputData = [
        'player_state',
        IOData('vel','array',array_size=[2]),
        IOData('task_position_offset','array',array_size=[2]),
        IOData('pos','array',array_size=[2])];
    inputData.append(IOData('collision_grid','array',[15,15]))
    
    runConfig = RunnerConfig(
        getFitness,
        getRunning,
        logging=True,
        parallel=True,
        gameName=NAME,
        returnData=inputData,
        num_trials=1,
        queue_type="ray",
        pool_type="ray");
    runConfig.tile_scale = 2;
    runConfig.view_distance = 3.75;
    runConfig.task_obstruction_score = task_obstruction_score;
    runConfig.external_render = False;
    runConfig.parallel_processes = 3;
    runConfig.chunkFactor = 24;
    runConfig.saveFitness = False;

    run_name = 'play_test'
    checkpoint_run_name = "run_10";

    runConfig.logPath = f'logs/smb1Py/run-{run_name}-log.txt';
    runConfig.fitness_collection_type='delta_max';

    if not local_display:
        runConfig.pool_kwargs = {'ray_remote_args':{'scheduling_strategy':st,'num_cpus':1}};
        
    gamerunner = GameRunner(game,runConfig);


    ### LOAD REPLAY RENDERER ###
    replay_runConfig = copy.deepcopy(runConfig)
    replay_runConfig.logging = False;
    replay_runConfig.training_data = TrainingDataManager("smb1Py","replay");
    replay_runConfig.reporters = [];
    replay_game = copy.deepcopy(game);
    replay_game.initInputs.update({"window_title":"Best Genomes - Instant Replay"})
    replay = ReplayRenderer.remote(replay_game,checkpoint_run_name,replay_runConfig)


    ### LOAD TASK FITNESS REPORTER ###
    fitness_save_path = Path("memories")/"smb1Py"/f"{run_name}_fitness_history";
    fitness_reporter = TaskFitnessReporter(save_path=fitness_save_path,queue_type="ray");
    game.register_reporter(fitness_reporter);

    ### LOAD FIXED NET ###

    tuned_modelsfolder = Path("models/tuning");

    model_path = 'models/test_q3_long3.model';
    tunedmodels = sorted(os.listdir(tuned_modelsfolder)) if os.path.exists(tuned_modelsfolder) else None;
    model_path = tuned_modelsfolder/tunedmodels[-1] if tunedmodels else model_path;
    model = None;
    with open(model_path,'rb') as f:
        model = torch.load(f,map_location=torch.device('cpu'));
    model.eval();

    model_grids = 'collision_grid';
    model_endpadding = (6,6);
    model_mindist,model_maxdist = 8,40
    
    #### LEVEL PLAYER SOURCE ####

    search_resolution = 4

    ### LOAD/GENERATE LEVEL ###

    level_path = Path('levels')/'smb1_levels'/'1-1.lvl';
    level = None;
    if (os.path.exists(level_path)):
        level = SegmentState(None,None,file_path=level_path);
    else:
        options = GenerationOptions(size=(50,15),inner_size=(48,13),num_blocks=(25,50),ground_height=(9,12),valid_start_blocks=(0,6));
        level:SegmentState = SegmentGenerator.generate(options)[0];
        level.task_bounds = None;
        level.task = (48*c.TILE_SIZE,10*c.TILE_SIZE);
        level.save_file(level_path);

    try:
        goalx = level.static_data[c.MAP_MAPS][0][c.MAP_FLAGX];
    except:
        goalx = 48*c.TILE_SIZE

    goals = [(goalx,i*c.TILE_SIZE) for i in range(20)]; 
    log(goals);

    save = "level_routing_checkpoints/level1-1.chp"
    checkpoint = None;
    if os.path.exists(save):
        log("loading checkpoint...")
        with open(save,'rb') as f:
            checkpoint = pickle.load(f);
        log("checkpoint successfully loaded")


    player = LevelPlayer();
    player.set_fixed_net(model,model_grids,model_endpadding,model_mindist,model_maxdist);
    player.set_level_info(int(runConfig.view_distance),runConfig.tile_scale);

    levelTDSource = player.play_level(
            game,
            level,
            goals,
            fitness_reporter,
            search_data_resolution=search_resolution,
            task_offset_downscale=2,
            search_checkpoint=checkpoint,
            checkpoint_save_location=save,
            training_dat_per_gen=30,
            fitness_aggregation_type="q3",
            replay_renderer=replay);
    




    ### AUTO GENERATED TRAINING SOURCE ###

    configs = [
        GenerationOptions(num_blocks=0,ground_height=7,valid_task_blocks=c.FLOOR,valid_start_blocks=c.FLOOR), #0
        GenerationOptions(num_blocks=0,ground_height=7,valid_task_blocks=c.INNER,valid_start_blocks=c.FLOOR), #1
        GenerationOptions(num_blocks=(1,3),ground_height=7,valid_task_blocks=c.INNER,valid_start_blocks=c.FLOOR), #2
        GenerationOptions(num_blocks=(0,4),ground_height=7,task_batch_size=(1,4)), #3
        GenerationOptions(num_blocks=(0,8),ground_height=(7,8),task_batch_size=(1,4)), #4
        GenerationOptions(num_blocks=(0,4),ground_height=7,task_batch_size=(1,4),num_gaps=(1,2),gap_width=(1,2)), #5
        GenerationOptions(num_blocks=(0,6),ground_height=7,task_batch_size=(1,4),num_gaps=(1,2),gap_width=(0,4)), #6
        GenerationOptions(num_blocks=(0,4),ground_height=7,task_batch_size=(1,4),num_gaps=(1,2),gap_width=(1,3),allow_gap_under_start=True), #7
        GenerationOptions(num_blocks=(0,6),ground_height=7,task_batch_size=(1,3),num_enemies={c.ENEMY_TYPE_GOOMBA:1},valid_enemy_positions=c.GROUNDED), #8
        GenerationOptions(size=(20,15),inner_size=(14,9),num_blocks=(0,8),ground_height=(7,8),task_batch_size=(1,4)), #9
        GenerationOptions(size=(18,14),inner_size=(12,8),num_blocks=(0,6),ground_height=7,task_batch_size=(1,4),num_gaps=(1,3),gap_width=(1,3)), #10
        ];
    
    orders = [(configs[4],15),(configs[2],5),(configs[6],4),(configs[7],1),(configs[9],10),(configs[10],5)];

    tdat_gen = partial(generate_data,orders);

    auto_gen_source = GeneratorTDSource(tdat_gen);



    ### INITIALIZE TDMANAGER ###
    manager = SourcedShelvedTDManager[SegmentState]('smb1Py',run_name,initial_sources=[levelTDSource,auto_gen_source]);
    runConfig.training_data = manager;


    ### CREATE MODEL TUNER ###

    torch_device_type = "cuda" if torch.cuda.is_available() else "cpu";
    model_lr = 5e-6;
    model_epochs_per_gen = 5;
    model_batchsize = 20;

    model_tuner = ModelTunerReporter(
        model,
        model_grids,
        model_endpadding,
        model_mindist,model_maxdist,
        int(runConfig.view_distance),runConfig.tile_scale,
        search_resolution,
        fitness_reporter,
        manager,
        model_lr, model_epochs_per_gen, model_batchsize,
        device_type=torch_device_type,
        fitness_aggregation_type="q3",
        model_save_path=tuned_modelsfolder,
        model_save_format="run10_model_gen{0}.model");

    runConfig.reporters.append(model_tuner);
    runConfig.reporters.append(fitness_reporter);
    runConfig.reporters.append(player);

    ### RUN NEAT ###
    replay.start_replay_loop.remote();

    gamerunner.continue_run(checkpoint_run_name);
    
    
    log("Level successfully completed!! Winning Path:",player.winning_path,"completed using the population of generation",player.gamerunner.generation);
    

