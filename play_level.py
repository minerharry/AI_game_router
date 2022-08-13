import copy
import itertools
import math
from typing import DefaultDict, Iterable, Literal
from baseGame import EvalGame
from gameReporting import ThreadedGameReporter
from game_runner_neat import GameRunner
from games.smb1Py.py_mario_bros.PythonSuperMario_master.smb_game import SMB1Game
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
import torch
import numpy as np
from matplotlib import pyplot as plt
from runnerConfiguration import RunnerConfig

from search import DStarSearcher, LevelSearcher
from smb1Py_runner import NAME, getFitness, getRunning, task_obstruction_score
from training_data import TrainingDataManager

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


class LevelPlayer:
    
    def __init__(self): pass;

    def set_NEAT_player(self,game:EvalGame,runConfig:RunnerConfig,run_name:str,player_view_distance:None|int=None,player_tile_scale:None|int=None,config_override=None):
        self.runConfig = runConfig;
        self.neat_config_override = config_override;
        self.run_name = run_name;

        self.tdat = TrainingDataManager[SegmentState]('smb1Py',run_name);
        self.runConfig.training_data = self.tdat;
        
        self.game = game;
        self.runConfig.generations = 1;
        self.gamerunner = GameRunner(self.game,self.runConfig);

        self.task_reporter = TaskFitnessReporter();
        self.game.register_reporter(self.task_reporter);

        self.view_distance = player_view_distance if player_view_distance else self.runConfig.view_distance;
        self.tile_scale = player_tile_scale if player_tile_scale else self.runConfig.tile_scale;


    def set_fixed_net(self,model:torch.nn.Module,used_grids:str|list[str],endpoint_padding:tuple[int,int],minimum_viable_distance:float,maximum_viable_distance:float):
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

    def eval_fixed_net(self,grids,start:tuple[int,int],task:tuple[int,int],size:tuple[int,int])->float:
        x_grad = np.array([[f_x_grad(x,y,start,task) for x in range(size[0])] for y in range(size[1])]);
        y_grad = np.array([[f_y_grad(x,y,start,task) for x in range(size[0])] for y in range(size[1])]);
        smear = np.array([[f_smear(x,y,start,task) for x in range(size[0])] for y in range(size[1])]);

        grids += [x_grad,y_grad,smear];
        
        full = np.stack(grids,axis=0);
        return self.fixed_net(full);

    def eval_NEAT_player(self,tasks:list[list[tuple[float,float]]],level:SegmentState)->Iterable[list[tuple[tuple[tuple[float,float],tuple[float,float]],float]]]:
        data = [];
        for task_path in tasks:
            state = copy.deepcopy(level);
            state.task = task_path[0];
            state.task_path = task_path;
            data.append(state);

        self.tdat.set_data(data);

        self.gamerunner.continue_run(self.run_name,manual_config_override=self.neat_config_override);

        return self.task_reporter.get_all_data();


    #given a level (no task, no task_bounds), a goal block or set of blocks
    def play_level(self,level:SegmentState,goal:tuple[float,float]|list[tuple[float,float]],training_dat_per_gen=50,search_data_resolution=4):
                
        ### Extract level info for routing purposes
        game = tools.Control()
        state_dict = {c.LEVEL: Segment()}
        game.setup_states(state_dict, c.LEVEL)
        game.state.startup(0,{c.LEVEL_NUM:1},initial_state=level);

        gdat = game.get_game_data(self.view_distance,self.tile_scale);
        mdat = game.get_map_data(search_data_resolution);

        player_start = gdat['pos'];
        search_grids = [np.array(mdat[g]) for g in self.fixed_grids];
        grid_size = search_grids[0].shape
        grids_bounds:tuple[int,int,int,int] =  mdat['grid_bounds'];

        if not isinstance(goal,list):
            goal = [goal];

        def pos_to_grid_index(pos:tuple[float,float]):
            position_parameter = ((pos[0]-grids_bounds[0])/(grids_bounds[1]-grids_bounds[0]),(pos[1]-grids_bounds[2])/(grids_bounds[3]-grids_bounds[2]));
            closest_pixel = (int(position_parameter[0]*grid_size[0]),int(position_parameter[1]*grid_size[1]))
            return closest_pixel;

        
        ### initialize fixed net and cost dict
        def get_cost(start:tuple[int,int],task:tuple[int,int]):
            if (start,task) in costs:
                return costs[start,task];
            return self.eval_fixed_net(search_grids,start,task,grid_size);
            
        costs = dict[tuple[tuple[int,int],tuple[int,int]],float]();

        ### Initialize D* Searcher (run in reverse through the level)
        task_offsets = list(annulus(self.min_task_dist,self.max_task_dist));

        def d_star_heuristic(p1:tuple[int,int]|Literal['goal'],p2:tuple[int,int]|Literal['goal']): #made to be bidirectional since lazy
            if isinstance(p1,str):
                return min([d_star_heuristic(pos,p2) for pos,_ in d_star_succ(p1)]);
            if isinstance(p2,str):
                return d_star_heuristic(p2,p1);

            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5;

        def d_star_pred(pos:tuple[int,int]|Literal['goal']): #since d* in reverse, this actually gives successors. returns list of (succ,cost(pos,succ))
            if (isinstance(pos,str)):
                raise Exception("cannot find predecessors of the beginning");

            for offset in task_offsets:
                task = (pos[0]+offset[0],pos[1]+offset[1]);
                yield (task,get_cost(pos,task));
            
            if pos in goal_idxs:
                yield ('goal',0);

        def d_star_succ(pos:tuple[int,int]|Literal['goal']): #since d* in reverse, this actually gives successors. returns list of (pred,cost(pred,pos))
            if (isinstance(pos,str)):
                return goal_idxs;

            for offset in task_offsets:
                task = (pos[0]+offset[0],pos[1]+offset[1]);
                yield (task,get_cost(pos,task));

        start_idx = pos_to_grid_index(player_start);
        goal_idxs = [pos_to_grid_index(g) for g in goal];

        d_searcher = DStarSearcher[tuple[int,int]|Literal['goal']](d_star_heuristic,start='goal',goal=start_idx,pred=d_star_pred,succ=d_star_succ);

        ### Initialize A* Searcher (runs the level forward, selects training data)

        d_g = None;

        def a_star_heuristic(path:list[tuple[int,int]]):
            return d_g[path[-1]];

        def a_star_succ(path:list[tuple[int,int]]):
            end = path[-1];
            for offset in task_offsets:
                new = (end[0]+offset[0],end[1]+offset[1]);
                newpath = path.copy();
                newpath.append(new);
                yield newpath;

        def a_star_cost(p1:list[tuple[int,int]],p2:list[tuple[int,int]]):
            return get_cost(p1[-1],p2[-1]);

        a_searcher = LevelSearcher[list[tuple[int,int]],tuple[int,int]](player_start,lambda x: x[-1] in goal_idxs,a_star_heuristic,lambda node: node[-1],a_star_succ,a_star_cost);

        ### ROUTE + PLAY LEVEL
        
        level_finished = False;

        updates:list[tuple[tuple[int,int]|Literal['goal'],tuple[int,int]|Literal['goal'],float,float]] = [];

        def scan_cost_updates():
            return updates;

        d_iter = d_searcher.search_iter(scan_cost_updates);
        d_g = next(d_iter).g;

        while not level_finished:
            top_edges = a_searcher.sorted_edges()[:training_dat_per_gen];
                        
            fitnesses = self.eval_NEAT_player(level,top_edges);  # type: ignore

            updates = [];
            a_updates = [];
            for fitness_list in fitnesses:
                acc_fitness = 0;
                for (start,end),fitness in fitness_list:
                    start = pos_to_grid_index(start);
                    end = pos_to_grid_index(start);

                    old_cost = costs[start,end];
                    costs[start,end] = fitness;
                    updates.append((start,end,old_cost,fitness));

                    acc_fitness += fitness;
                    a_updates.append((end,acc_fitness));

            d_g = next(d_iter).g; #generator syntax; makes d_searcher update the next round of searching, pulling from the updates list as updates to its costs.

            a_searcher.update_scores(a_updates);




        

class TaskFitnessReporter(ThreadedGameReporter[list[tuple[tuple[tuple[float,float],tuple[float,float]],float]]]): #I'm so sorry
    
    def on_start(self, game: SMB1Game):
        self.previous_task = game.getMappedData()['pos'];
        self.current_task = game.getMappedData()['task'];
        self.current_fitness = game.getFitnessScore();
        self.current_data:list[tuple[tuple[tuple[float,float],tuple[float,float]],float]] = [];

    def on_tick(self, game: SMB1Game, inputs, finish = False):
        task = game.getMappedData()['task'];
        if (task != self.current_task or finish):
            out_fitness = game.getFitnessScore() - self.current_fitness;
            self.current_data.append(((self.previous_task,self.current_task),out_fitness));
            self.previous_task = self.current_task;

            self.current_task = task;
            self.current_fitness = game.getFitnessScore();

    def on_finish(self, game: SMB1Game):
        self.on_tick(game,None,finish=True);
        self.put_data(self.current_data);


    
if __name__== "__main__":
    player = LevelPlayer();
    game = EvalGame(SMB1Game);
    runConfig = RunnerConfig(
        getFitness,
        getRunning,
        logging=True,
        parallel=True,
        gameName=NAME,
        returnData=inputData,
        num_trials=1);
    runConfig.tile_scale = 2;
    runConfig.view_distance = 3.75;
    runConfig.task_obstruction_score = task_obstruction_score;
    runConfig.external_render = False;
    runConfig.parallel_processes = 6;
    runConfig.chunkFactor = 24;
    runConfig.saveFitness = True;

    run_name = 'play_test'

    runConfig.logPath = f'logs\\smb1Py\\run-{run_name}-log.txt';
    runConfig.fitness_collection_type='delta';
    player.set_NEAT_player(game,runConfig,run_name,runConfig.view_distance,runConfig.tile_scale);

    model_path = ''
    model = None;
    with open(model_path,'rb') as f:
        model = torch.load(f);
    player.set_fixed_net(model,'collision_grid',(6,6),)