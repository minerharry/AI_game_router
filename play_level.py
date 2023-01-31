from __future__ import annotations
import argparse
from functools import lru_cache, partial
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import time
from typing import Callable, DefaultDict, Generic, Iterable, Literal, NamedTuple, TypeVar
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

from level_renderer import LevelRenderer, LevelRendererReporter

from neat.reporting import BaseReporter

import torch
from torch import Tensor
from torch.utils.data import DataLoader, random_split

import numpy as np
from tqdm import tqdm

try:
    import ray
    from ray_event import RayEvent
except:
    ray = None
    
from runnerConfiguration import IOData, RunnerConfig
from search import DStarSearcher, LevelSearcher
from smb1Py_runner import NAME, generate_data, getFitness, getRunning, task_obstruction_score
from training_data import GeneratorTDSource, IteratorTDSource, SourcedShelvedTDManager,TDSource, TrainingDataManager


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
                a_checkpoint:dict):
        self.costs = edge_costs;
        self.pred = edge_predictions;
        self.d = d_checkpoint;
        self.a = a_checkpoint;


class LevelPlayer:
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

    def log(self,*args,**kwargs):
        print(*args,**kwargs);

    def get_checkpoint_data(self):
        return LevelCheckpoint(self.costs,self.predictions,self.d_searcher.get_checkpoint_data(),self.a_searcher.get_checkpoint_data());

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
            multiprocessing_type="ray")->TDSource[SegmentState]:

        gen = self._yield_NEAT_data(game,level,goal,fitness_reporter,training_dat_per_gen,search_data_resolution,task_offset_downscale,search_checkpoint,checkpoint_save_location,fitness_aggregation_type,render_progress,multiprocessing_type)
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
            multiprocessing_type:Literal["ray","multiprocessing"]):
        self.multi = multiprocessing_type
        if self.multi not in ["multiprocessing","ray"]:
            raise TypeError(f"library type {self.multi} not known");

        self.game = game;
        self.task_reporter = fitness_reporter
        # self.task_reporter.add_capture_source(self.source);

        fitness_from_list = {
            "max":max,
            "median":np.median,
            "mean":np.average,
            "q3":lambda l: np.quantile(l,0.75),
            "q1":lambda l: np.quantile(l,0.25),
            "min":min
        }[fitness_aggregation_type];
        log = self.log #shorthand

        if search_data_resolution % task_offset_downscale != 0:
            print("WARNING: Attempting to downscale task resolution by a nonfactor of the data resolution. Setting task resolution to data resolution.")
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
                    print(p1)
                    print(list(d_star_succ(p1)));
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



        while not level_finished:
            log("checkpoint saved")
            ##save checkpoint
            save_data = self.get_checkpoint_data();
            with open(checkpoint_save_location,'wb') as f:
                pickle.dump(save_data,f);


            log("retrieving best edges from A*")
            top_paths = self.a_searcher.sorted_edges()
            
            # print('top ten scores:',[(self.a_searcher.sort_key(e),e) for e in top_paths[:10]]);

            top_paths = top_paths[:training_dat_per_gen];
            log(len(top_paths),"edges retrieved");

            player_paths = [[grid_index_to_pos(task) for task in path] for _prev,path in top_paths];
            

            log("NEAT-player attempting level segments");
            

            ### Evaluating neat player, used to be its own function, copy-pasted
            tasks = player_paths

            self.log("evaluating NEAT-player with training data",len(tasks),'ex:',tasks[0],'and level',level);
            data:list[SegmentState] = [];
            for i,task_path in enumerate(tasks):
                state = level.deepcopy();
                task_path = task_path[1:];
                state.task = task_path[0];
                state.task_path = task_path;
                state.source_id = i;
                data.append(state);
            

            renderProcess = None;
            if self.renderer is not None:
                reached_idxs = [p[1][-1] for p in self.a_searcher.completed_edges];
                failed_idxs = [p for _,p in self.costs.keys() if p not in reached_idxs];

                reached = [self.grid_index_to_pos(idx) for idx in reached_idxs];
                failed = [self.grid_index_to_pos(idx) for idx in failed_idxs];
                paths:dict[int,Iterable[tuple[float,float]]] = {state.source_id:[self.player_start,state.task] + state.task_path for state in data}; #type: ignore

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

            yield data;


            result:list[list[tuple[tuple[floatPos, floatPos | Literal['complete']], float]]] = [d.data[1] for d in self.task_reporter.get_all_conserved_data() if d.data[0] == self.source];
            
            print("Neat player evaluated;",len(result),"data collected");

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
            
            for fitness_list in all_fitnesses:
                acc_fitness = 0;
                acc_path:list[gridPos] = [];
                for (start,end),fitness in fitness_list:
                    if end == 'complete':
                        if tuple(acc_path) not in completed_paths:
                            completed_paths.append(tuple(acc_path)) 
                            if (pos_to_grid_index(start) in goal_idxs):
                                level_finished = True;
                                winning_path = acc_path;

                    else:
                        start = pos_to_grid_index(start);
                        end = pos_to_grid_index(end);
                        if start == end:
                            print("ERROR: start and end should not be the same; from path",fitness_list)
                            continue;

                        best_segments[start,end].append(fitness);

                        acc_fitness += fitness;

                        if len(acc_path) == 0:
                            acc_path.append(start);
                        acc_path.append(end);
                        
                        best_paths[tuple(acc_path)].append(acc_fitness);

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
        print("winning path found! Path:",self.winning_path);

task_out_k = list[tuple[tuple[floatPos,floatPos|Literal['complete']],float]]#I'm so sorry
class TaskFitnessReporter(BaseReporter,ThreadedGameReporter[IdData[tuple[TDSource,task_out_k]]]): 
    def __init__(self,capture_sources:Iterable[TDSource]|None=None,save_path=None,**kwargs):
        super().__init__(**kwargs);
        self.save_path = Path(save_path) if save_path else None;
        self.generation = None;
        # self.captures:list[TDSource] = list(capture_sources or []);
        # self.source_id_map:dict[TDSource,set[int]] = DefaultDict(lambda:set());
        self.data_list:list[IdData[tuple[TDSource,task_out_k]]]|None = None
        self.data = None;
        self.data_source:TDSource|None = None;

    # def add_capture_source(self,*source:TDSource):
    #     self.captures.extend(source);
    
    # def remove_capture_source(self,*source:TDSource):
    #     [self.captures.remove(s) for s in source];


    #NOTE: EXECUTED IN PARALLEL PROCESSES
    def on_training_data_load(self, game: SMB1Game, id:int):
        if self.data_list is not None:
            self.put_all_data(*self.data_list);
        self.data_list = [];
        if self.data_id != id:
            self.data_id = id;
            t:SourcedShelvedTDManager = game.runConfig.training_data #type: ignore
            s = t.get_datum_source(id)
            self.data_source = s;

    def on_start(self, game: SMB1Game):
        self.previous_task:floatPos = game.getMappedData()['pos'];
        self.current_task:floatPos = game.getMappedData()['task_position'];
        self.current_fitness = game.getFitnessScore();
        self.current_data:list[tuple[tuple[floatPos,floatPos|Literal['complete']],float]] = [];

    def on_tick(self, game: SMB1Game, inputs, finish = False):
        task:floatPos = game.getMappedData()['task_position'];
        if (task != self.current_task or finish):
            out_fitness = game.getFitnessScore() - self.current_fitness;
            self.current_data.append(((self.previous_task,self.current_task),out_fitness));
            self.previous_task = self.current_task;

            self.current_task = task;
            self.current_fitness = game.getFitnessScore();

    def on_finish(self, game: SMB1Game):
        self.on_tick(game,None,finish=True);
        if game.getMappedData()['task_path_complete']:
            self.current_data.append(((self.previous_task,'complete'),-1));
        assert self.data_list is not None
        assert self.data_source is not None
        self.data_list.append(IdData(self.data_id,(self.data_source,self.current_data)));

    def start_generation(self, generation):
        print(f"Task Fitness Reporter - generation {generation} started");
        self.generation = generation;
        super().get_all_data(); #flush data contents
        self.data = None;
        self.source_id_map = DefaultDict(lambda:set());

    def end_generation(self, config, population, species_set):
        print(f"Task Fitness Reporter - generation {self.generation} ended");
        if self.generation is not None and self.save_path:
            data = list(self.get_all_conserved_data());
            out:dict[int,list[list[float]]] = DefaultDict(lambda: []);
            for d in data:
                out[d.id].append([f[1] for f in d.data[1]]);
            
            out_path = self.save_path/f"gen_{self.generation}";
            checkpoint = FitnessCheckpoint(out); #type: ignore
            checkpoint.save_checkpoint(out_path);

    def get_all_data(self):
        raise NotImplementedError("unable to pull all data from fitness reporter; if you want to pull data from a source, add a capture source");

    def get_all_conserved_data(self):
        if self.data is not None:
            return self.data
        out:list[IdData[tuple[TDSource,task_out_k]]] = [];
        for d in list(super().get_all_data()):
            out.append(d);
            self.put_data(d);
        self.data = out;
        return out;


    # def get_captured_data(self,capture_source:TDSource):
    #     if capture_source not in self.captures:
    #         raise ValueError(f"capture source {capture_source} not tracked by reporter!")
    #     out:list[IdData[task_out_k]] = [];
        
    #     captured_ids = self.source_id_map[capture_source];
    #     for d in list(self.get_all_conserved_data()):
    #         if d.id in captured_ids:
    #             out.append(d);

    #     return out;

def train_loop(dataloader:DataLoader, model, loss_fn, optimizer,weight_fun:Callable[[Tensor,Tensor,Tensor],Tensor]|None=None):
    loss = None;
    for (X,y) in tqdm(dataloader,leave=False):
        # Compute prediction and loss
        pred = model(X)
        # print(X.shape);
        # print(y.shape);
        # print(pred.shape);

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
            # print(X.shape);
            pred = model(X)
            # print(y);
            if weight_fun:
              loss = loss_fn(pred, y, weight_fun(X,y,pred));
            else:
              loss = loss_fn(pred,y);
            test_loss += loss.item()
            # correct += np.array(pred-y).mean();

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
            model_save_format:str="model_gen{0}.model"):
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

    def get_search_grids(self,td_id:int):
        level = self.manager[td_id];

        data_game = Segment()
        data_game.startup(0,{c.LEVEL_NUM:1},initial_state=level);

        print("acquiring map data")
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
        all_data = self.reporter.get_all_conserved_data();
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

        grids = np.expand_dims(grids,0);

        return Tensor(grids);
        

    def update_model(self,data:list[IdData[tuple[TDSource,task_out_k]]]):
        print("Tuning Model from generation's fitnesses");

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

                input_grids = self.get_input_grids(gridstart,gridend,grid_size,search_grids);

                d = dist(*start,task);

                shaped_grids[input_grids.shape[1:]].append((input_grids.to(self.device_type),Tensor([fitness/d]).to(self.device_type)));
        
        sets = [random_split(s,[len(s)-int(len(s)*self.test_fraction),int(len(s)*self.test_fraction)], generator=torch.Generator().manual_seed(0)) for s in shaped_grids.values()];

        train_dataloader = [b for loader in [DataLoader(data[0],batch_size=self.batch_size) for data in sets] for b in loader];
        test_dataloader = [b for loader in [DataLoader(data[1],batch_size=self.batch_size) for data in sets] for b in loader ];

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,nesterov=True);

        for t in tqdm(range(self.epochs),desc="Model Fine-Tuning: "):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer)
            test_loop(test_dataloader, model, loss_fn)

        print("Model Tuned!");

    
        

                


        
    
if __name__== "__main__":
    
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

    print("Displaying locally" if local_display else "Displaying remotely");

    

    if not local_display:
        ray.init(address=ip);
        print("waiting for display node...");
        num_display = 0
        while num_display < 2:
            r = ray.cluster_resources();
            if "Display" in r:
                num_display = r["Display"]
            time.sleep(5);
        print("display node obtained, display cores available:",num_display);
        print("cluster nodes:",ray.nodes());

        basic_cores = ray.cluster_resources()["CPU"]-num_display-2; #two extra cores for whatever

        cpu_bundles = [{"CPU":1.0} for _ in range(int(basic_cores))];
        display_bundles = [{"Display":0.01,"CPU":1} for _ in range(int(num_display) - 1)];

        total_bundles = cpu_bundles + display_bundles
        group = placement_group(total_bundles,strategy="SPREAD");
        ray.get(group.ready());
        print(placement_group_table(group));
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
    runConfig.parallel_processes = 10;
    runConfig.chunkFactor = 24;
    runConfig.saveFitness = False;

    run_name = 'play_test'

    runConfig.logPath = f'logs/smb1Py/run-{run_name}-log.txt';
    runConfig.fitness_collection_type='delta_max';

    if not local_display:
        runConfig.pool_kwargs = {'ray_remote_args':{'scheduling_strategy':st,'num_cpus':1}};
        
    gamerunner = GameRunner(game,runConfig);


    ### LOAD TASK FITNESS REPORTER ###
    fitness_save_path = Path("memories")/"smb1Py"/f"{run_name}_fitness_history";
    fitness_reporter = TaskFitnessReporter(save_path=fitness_save_path,queue_type="ray");


    ### LOAD FIXED NET ###

    model_path = 'models/test_q3_long3.model'
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
    print(goals);

    save = "level_routing_checkpoints/level1-1.chp"
    checkpoint = None;
    if os.path.exists(save):
        print("loading checkpoint...")
        with open(save,'rb') as f:
            checkpoint = pickle.load(f);
        print("checkpoint successfully loaded")


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
            training_dat_per_gen=40,
            fitness_aggregation_type="q3");
        




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
    
    orders = [(configs[4],45),(configs[2],15),(configs[6],10),(configs[7],5),(configs[9],30),(configs[10],15)];

    tdat_gen = partial(generate_data,orders);

    auto_gen_source = GeneratorTDSource(tdat_gen);


    ### INITIALIZE TDMANAGER ###
    manager = SourcedShelvedTDManager[SegmentState]('smb1Py',run_name,initial_sources=[levelTDSource,auto_gen_source]);
    # manager.add_source();
    runConfig.training_data = manager;


    ### CREATE MODEL TUNER ###

    torch_device_type = "cuda" if torch.cuda.is_available() else "cpu";
    model_lr = 1e-5;
    model_epochs_per_gen = 10;
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
        model_save_path="models/tuning",
        model_save_format="run10_model_gen{0}.model");

    runConfig.reporters.append(model_tuner);

    ### RUN NEAT ###
    gamerunner.continue_run("run_10");
    
    
    # print("Level successfully completed!! Winning Path:",player.winning_path,"completed using the population of generation",player.gamerunner.generation);
    

