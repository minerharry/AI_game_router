import copy
import itertools
import math
import os
from pathlib import Path
import pickle
import random
from typing import Callable, DefaultDict, Iterable, Literal
from baseGame import EvalGame
from gameReporting import ThreadedGameReporter
from game_runner_neat import GameRunner
from games.smb1Py.py_mario_bros.PythonSuperMario_master.smb_game import SMB1Game
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segmentGenerator import GenerationOptions, SegmentGenerator
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from runnerConfiguration import IOData, RunnerConfig
from guppy import hpy
hp = hpy();

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


class LevelCheckpoint:
    def __init__(self,
                edge_costs:dict[tuple[tuple[int,int],tuple[int,int]],float],
                edge_predictions:dict[tuple[tuple[int,int],tuple[int,int]],float],
                d_checkpoint:dict,
                a_checkpoint:dict):
        self.costs = edge_costs;
        self.pred = edge_predictions;
        self.d = d_checkpoint;
        self.a = a_checkpoint;

class LevelPlayer:
    
    def __init__(self): pass;

    def set_NEAT_player(self,game:EvalGame,runConfig:RunnerConfig,run_name:str,
            player_view_distance:None|int=None,
            player_tile_scale:None|int=None,
            config_override=None,
            checkpoint_run_name:str|None=None,
            extra_training_data:Iterable[SegmentState]|None=None,
            extra_training_data_gen:Callable[[],Iterable[SegmentState]]|None=None):

        self.runConfig = runConfig;
        self.neat_config_override = config_override;
        self.run_name = run_name;
        self.checkpoint_run_name = checkpoint_run_name if checkpoint_run_name else self.run_name;

        self.tdat = TrainingDataManager[SegmentState]('smb1Py',run_name);
        self.runConfig.training_data = self.tdat;

        self.extra_dat_gen = extra_training_data_gen if extra_training_data_gen else ((lambda: extra_training_data) if extra_training_data else None);
        
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
        self.fixed_padding = endpoint_padding;

    def eval_fixed_net(self,grids,start:tuple[int,int],task:tuple[int,int],size:tuple[int,int])->float:
        
        left,right = (start[0],task[0]) if start[0] < task[0] else (task[0],start[0]);
        top,bottom = (start[1],task[1]) if start[1] < task[1] else (task[1],start[1]);

        #left,right,top,bottom
        bounds = [max(left-self.fixed_padding[0],0),min(right+self.fixed_padding[0],size[0]-1),max(top-self.fixed_padding[1],0),min(bottom+self.fixed_padding[1],size[1]-1)];
        x_grad = np.array([[f_x_grad(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();
        y_grad = np.array([[f_y_grad(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();
        smear = np.array([[f_smear(x,y,start,task) for x in range(bounds[0],bounds[1]+1)] for y in range(bounds[2],bounds[3]+1)]).transpose();

        grids = np.array(grids)[:,bounds[0]:bounds[1]+1,bounds[2]:bounds[3]+1]
        
        grids:np.ndarray = np.concatenate((grids,[x_grad,y_grad,smear]),axis=0)

        grids = np.expand_dims(grids,0);

        full = torch.Tensor(grids);
        return self.fixed_net(full).item();

    def eval_NEAT_player(self,tasks:list[list[tuple[float,float]]],level:SegmentState):
        '''returns the fitness values (for ALL players) for each leg of the task'''
        self.log("evaluating NEAT-player with training data",len(tasks),'ex:',tasks[0],'and level',level);
        data = [];
        for task_path in tasks:
            state = copy.deepcopy(level);
            task_path = task_path[1:];
            state.task = task_path[0];
            state.task_path = task_path;
            data.append(state);

        if self.extra_dat_gen:
            data += self.extra_dat_gen();

        self.tdat.set_data(data);

        print("evaluating on data:",list(zip(tasks,self.tdat.active_data.keys())));

        self.gamerunner.continue_run(self.checkpoint_run_name,manual_config_override=self.neat_config_override);

        return self.task_reporter.get_all_data();

    def log(self,*args,**kwargs):
        print(*args,**kwargs);

    def get_checkpoint_data(self):
        return LevelCheckpoint(self.costs,self.predictions,self.d_searcher.get_checkpoint_data(),self.a_searcher.get_checkpoint_data());

    #given a level (no task, no task_bounds), a goal block or set of blocks
    #offset downscale means with how much resolution are potential task blocks generated.
    def play_level(self,
            level:SegmentState,
            goal:tuple[float,float]|list[tuple[float,float]],
            training_dat_per_gen=50,
            search_data_resolution=5,
            task_offset_downscale=2,
            search_checkpoint:LevelCheckpoint|None=None,
            checkpoint_save_location="play_checkpoint.chp",
            fitness_aggregation_type="max"):

        fitness_from_list = {
            "max":max,
            "median":np.median,
            "mean":np.average,
            "q3":lambda l: np.quartile(l,0.75),
            "q1":lambda l: np.quartile(l,0.25),
            "min":min
        }[fitness_aggregation_type];
        log = self.log #shorthand

        if search_data_resolution % task_offset_downscale != 0:
            print("WARNING: Attempting to downscale task resolution by a nonfactor of the data resolution. Setting task resolution to data resolution.")
            task_offset_downscale = 1;

        ### Extract level info for routing purposes
        log("--EXTRACTING LEVEL INFO--");
        log("game startup");
        game = tools.Control()
        state_dict = {c.LEVEL: Segment()}
        game.setup_states(state_dict, c.LEVEL)
        game.state.startup(0,{c.LEVEL_NUM:1},initial_state=level);

        log("acquiring map data")
        gdat = game.get_game_data(self.view_distance,self.tile_scale);
        mdat = game.get_map_data(search_data_resolution);

        player_start:tuple[float,float] = gdat['pos'];
        search_grids = [np.array(mdat[g]) for g in self.fixed_grids];
        grid_size = search_grids[0].shape;
        grids_bounds:tuple[int,int,int,int] =  mdat['grid_bounds'];

        log('start position  (continuous):',player_start,'discrete grid size:',grid_size,'discrete grids\' bounds:',grids_bounds);

        if not isinstance(goal,list):
            goal = [goal];

        def in_bounds(pos:tuple[int,int]):
            return pos[0] >= 0 and pos[0] < grid_size[0] and pos[1] >= 0 and pos[1] < grid_size[1];

        def cost_from_fitness(fitness:float):
            if (fitness <= 0):
                return float('inf');
            return 1/fitness;

        def pos_to_grid_index(pos:tuple[float,float]):
            pos = pos[0]-c.TILE_SIZE/(2*search_data_resolution),pos[1]-c.TILE_SIZE/(2*search_data_resolution);
            position_parameter = ((pos[0]-grids_bounds[0])/(grids_bounds[1]-grids_bounds[0]),(pos[1]-grids_bounds[2])/(grids_bounds[3]-grids_bounds[2]));
            closest_pixel = (round(position_parameter[0]*grid_size[0]),round(position_parameter[1]*grid_size[1]))
            return closest_pixel;

        def grid_index_to_pos(index:tuple[int,int]):
            return (c.TILE_SIZE/(2*search_data_resolution)+index[0]/grid_size[0]*(grids_bounds[1]-grids_bounds[0])+grids_bounds[0],c.TILE_SIZE/(2*search_data_resolution)+index[1]/grid_size[1]*(grids_bounds[3]-grids_bounds[2])+grids_bounds[2]);

        
        if search_checkpoint is None:
            self.costs = dict[tuple[tuple[int,int],tuple[int,int]],float]();
            self.predictions = dict[tuple[tuple[int,int],tuple[int,int]],float]();
        else:
            self.costs = search_checkpoint.costs;
            self.predictions = search_checkpoint.pred;

        def get_cost(start:tuple[int,int],task:tuple[int,int]):
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

        def d_star_heuristic(p1:tuple[int,int]|Literal['goal'],p2:tuple[int,int]|Literal['goal']): #made to be bidirectional since lazy
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

        def d_star_succ(pos:tuple[int,int]|Literal['goal']): #returns list of (succ,cost(pos,succ))
            if (isinstance(pos,str)):
                raise Exception("cannot find successors of the goal");
            # log("d_preds");
            for offset in tqdm(task_offsets):
                task = (pos[0]+offset[0],pos[1]+offset[1]);
                if (in_bounds(task)):
                    yield (task,get_cost(pos,task));
            
            if pos in goal_idxs:
                yield ('goal',0);

        def d_star_pred(pos:tuple[int,int]|Literal['goal']): #returns list of (pred,cost(pred,pos))
            # log("d_succs");
            if pos == 'goal':
                for i in goal_idxs:
                    yield i,0;
                return;

            for offset in tqdm(task_offsets):
                task = (pos[0]+offset[0],pos[1]+offset[1]);
                if (in_bounds(task)):
                    yield (task,get_cost(task,pos));

        start_idx = pos_to_grid_index(player_start);
        goal_idxs = [pos_to_grid_index(g) for g in goal];

        log('routing from start',start_idx,'to goals',goal_idxs,'in level of gridded size',grid_size);

        check = search_checkpoint.d if search_checkpoint is not None else None;
        self.d_searcher = DStarSearcher[tuple[int,int]|Literal['goal']](d_star_heuristic,start=start_idx,goal='goal',pred=d_star_pred,succ=d_star_succ,checkpoint=check);

        ### Initialize A* Searcher (runs the level forward, selects training data)
        log("initializing a*")

        def a_star_heuristic(path:tuple[tuple[int,int],...]):
            self.d_searcher.explore_from_position(path[-1]);
            return self.d_searcher.rhs[path[-1]];

        def a_star_succ(path:tuple[tuple[int,int],]):
            end = path[-1];
            for offset in task_offsets:
                new = (end[0]+offset[0],end[1]+offset[1]);
                if not in_bounds(new):
                    continue;
                newpath = list(path);
                newpath.append(new);
                yield tuple(newpath);

        def a_star_cost(p1:tuple[tuple[int,int],...],p2:tuple[tuple[int,int],...]):
            return get_cost(p1[-1],p2[-1]);

        check = search_checkpoint.a if search_checkpoint is not None else None;
        self.a_searcher = LevelSearcher[tuple[tuple[int,int],...],tuple[int,int]]((start_idx,),lambda x: x[-1] in goal_idxs,a_star_heuristic,lambda node: node[-1],a_star_succ,a_star_cost,frustration_multiplier=0.02,checkpoint=check);

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

            top_paths = [[grid_index_to_pos(task) for task in path] for _prev,path in top_paths];
            

            log("NEAT-player attempting level segments");
            all_fitnesses = self.eval_NEAT_player(top_paths,level);
            log("fitnesses calculated")
            # print(all_fitnesses);

            best_segments:dict[tuple[tuple[int,int],tuple[int,int]],list[float]] = DefaultDict(lambda:[]);
            best_paths:dict[tuple[tuple[int,int],...],list[float]] = DefaultDict(lambda:[])
            completed_paths:list[tuple[tuple[int,int]]] = [];
            
            for fitness_list in all_fitnesses:
                acc_fitness = 0;
                acc_path:list[tuple[int,int]] = [];
                for (start,end),fitness in fitness_list:
                    if end == 'goal':
                        if tuple(acc_path) not in completed_paths:
                            completed_paths.append(tuple(acc_path)) 
                            if (start in goal_idxs):
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

            d_updates:list[tuple[tuple[int,int]|Literal['goal'],tuple[int,int]|Literal['goal'],float,float]] = [];
            a_updates:list[tuple[tuple[tuple[int,int],...],float]] = [];

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

            for path in completed_paths:
                self.a_searcher.complete_edge((tuple(path[:-1]),tuple(path)));

            log('running A* iteration')
            self.a_searcher.update_scores(a_updates);
        
        return winning_path;



class TaskFitnessReporter(ThreadedGameReporter[list[tuple[tuple[tuple[float,float],tuple[float,float]|Literal['goal']],float]]]): #I'm so sorry
    
    def on_start(self, game: SMB1Game):
        self.previous_task:tuple[float,float] = game.getMappedData()['pos'];
        self.current_task:tuple[float,float] = game.getMappedData()['task_position'];
        self.current_fitness = game.getFitnessScore();
        self.current_data:list[tuple[tuple[tuple[float,float],tuple[float,float]|Literal['goal']],float]] = [];

    def on_tick(self, game: SMB1Game, inputs, finish = False):
        task:tuple[float,float] = game.getMappedData()['task_position'];
        if (task != self.current_task or finish):
            out_fitness = game.getFitnessScore() - self.current_fitness;
            self.current_data.append(((self.previous_task,self.current_task),out_fitness));
            self.previous_task = self.current_task;

            self.current_task = task;
            self.current_fitness = game.getFitnessScore();

    def on_finish(self, game: SMB1Game):
        self.on_tick(game,None,finish=True);
        if game.getMappedData()['task_path_complete']:
            self.current_data.append(((self.previous_task,'goal'),-1));
        self.put_data(self.current_data);
    
if __name__== "__main__":
    player = LevelPlayer();
    game = EvalGame(SMB1Game);

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
        num_trials=1);
    runConfig.tile_scale = 2;
    runConfig.view_distance = 3.75;
    runConfig.task_obstruction_score = task_obstruction_score;
    runConfig.external_render = False;
    runConfig.parallel_processes = 4;
    runConfig.chunkFactor = 24;
    runConfig.saveFitness = True;

    run_name = 'play_test'

    runConfig.logPath = f'logs\\smb1Py\\run-{run_name}-log.txt';
    runConfig.fitness_collection_type='delta';
    player.set_NEAT_player(game,runConfig,run_name,runConfig.view_distance,runConfig.tile_scale,checkpoint_run_name='run_10');

    model_path = 'models/test_q3_long3.model'
    model = None;
    with open(model_path,'rb') as f:
        model = torch.load(f,map_location=torch.device('cpu'));
    model.eval();
    player.set_fixed_net(model,'collision_grid',(6,6),8,40);

    level_path = Path('levels')/'testing'/'test3.lvl';
    level = None;
    if (os.path.exists(level_path)):
        level = SegmentState(None,None,file_path=level_path);
    else:
        options = GenerationOptions(size=(50,15),inner_size=(48,13),num_blocks=(25,50),ground_height=(9,12),valid_start_blocks=(0,6));
        level:SegmentState = SegmentGenerator.generate(options)[0];
        level.task_bounds = None;
        level.task = (48*c.TILE_SIZE,10*c.TILE_SIZE);
        level.save_file(level_path);

    goals = [(48*c.TILE_SIZE,i*c.TILE_SIZE) for i in range(20)]; 


    save = "test_level_play_t6.chp"
    checkpoint = None;
    if os.path.exists(save):
        print("loading checkpoint...")
        with open(save,'rb') as f:
            checkpoint = pickle.load(f);
        print("checkpoint successfully loaded")

    winning_path = player.play_level(level,
        goals,
        search_data_resolution=4,
        task_offset_downscale=2,
        search_checkpoint=checkpoint,
        checkpoint_save_location=save,
        training_dat_per_gen=25);

