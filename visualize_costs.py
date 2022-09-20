from pathlib import Path
import pickle
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
from play_level import LevelCheckpoint, annulus;
import numpy as np

checkpoint_loc = "test_level_play_t3.chp";
level_path = Path('levels')/'testing'/'test3.lvl';

level = SegmentState(None,None,file_path=level_path);

with open(checkpoint_loc,'rb') as f:
    check:LevelCheckpoint = pickle.load(f);

preds = check.pred;
costs = check.costs;

game = tools.Control()
state_dict = {c.LEVEL: Segment()}
game.setup_states(state_dict, c.LEVEL)
game.state.startup(0,{c.LEVEL_NUM:1},initial_state=level);

search_data_resolution=4
task_offset_downscale=2
view_distance = 3.75;
tile_scale = 2;

gdat = game.get_game_data(view_distance,tile_scale);
mdat = game.get_map_data(search_data_resolution);

player_start:tuple[float,float] = gdat['pos'];
search_grids = [np.array(mdat[g]) for g in ['collision_grid']];
grid_size = search_grids[0].shape;
grids_bounds:tuple[int,int,int,int] =  mdat['grid_bounds'];

def get_cost(start:tuple[int,int],task:tuple[int,int]):
    if mdat['collision_grid'][task[0]][task[1]] != 0:
        return float('inf');
    if (start,task) in costs:
        return costs[start,task];
    elif (start,task) not in preds:
        raise Exception("Error: cannot evaluate fixed net outside of play conditions");
        self.predictions[start,task] = cost_from_fitness(dist(*grid_index_to_pos(start),grid_index_to_pos(task))*self.eval_fixed_net(search_grids,start,task,grid_size));
    return preds[start,task];

min_dist = 8; max_dist = 40
task_offsets = [(s[0]*task_offset_downscale,s[1]*task_offset_downscale) for s in annulus(min_dist/task_offset_downscale,max_dist/task_offset_downscale)]

def get_costs(pos)->np.ndarray:
    offset_array = np.ndarray((max_dist*2+1,max_dist*2+1))
    offset_array.fill(-1);
    center = (max_dist,max_dist);
    for offset in task_offsets:
        task = (pos[0]+offset[0],pos[1]+offset[1]);
        if (in_bounds(task)):
            offset_array[task[0]+center[0],task[1]+center[1]] = get_cost(pos,task);
    return offset_array;

def in_bounds(pos:tuple[int,int]):
    return pos[0] >= 0 and pos[0] < grid_size[0] and pos[1] >= 0 and pos[1] < grid_size[1];

def pos_to_grid_index(pos:tuple[float,float]):
    pos = pos[0]-c.TILE_SIZE/(2*search_data_resolution),pos[1]-c.TILE_SIZE/(2*search_data_resolution);
    position_parameter = ((pos[0]-grids_bounds[0])/(grids_bounds[1]-grids_bounds[0]),(pos[1]-grids_bounds[2])/(grids_bounds[3]-grids_bounds[2]));
    closest_pixel = (int(position_parameter[0]*grid_size[0]),int(position_parameter[1]*grid_size[1]))
    return closest_pixel;