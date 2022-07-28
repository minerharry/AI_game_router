from pathlib import Path
import pickle

import numpy as np
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
from training_data import TrainingDataManager
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
import json

training_data_folder = Path("memories/smb1Py");
dat:dict[int,dict[int,float]] = None;
p = training_data_folder/'run_10_top_15_fitness_history'/'gen_1530';
p = p.resolve();
print(p);
with open(p,'rb') as f:
    dat = pickle.load(f);
# print(len(dat));


print(len(dat));


TDM = TrainingDataManager[SegmentState]('smb1Py',10);

task_arrays:list[list[int]] = None;
with open(training_data_folder/'run-10-taskarray-data','r') as f:
    task_arrays = json.load(f);

game = tools.Control();
state_dict = {c.LEVEL: Segment()}
game.setup_states(state_dict, c.LEVEL)
started = False;
tile_scale = 2;
view_distance = 8*tile_scale-1;
for ids in task_arrays:
    task_fitness_output = {};
    if not started:
        game.state.startup(0,{c.LEVEL_NUM:1},initial_state=TDM[ids[0]]);
        started = True;
    else:
        game.load_segment(TDM[ids[0]]);
    gdat = game.get_game_data(view_distance,tile_scale);
    print(gdat);
    
    for id in ids:
        state = TDM[id];
        task_fitness_output[tuple(state.task)] = max(dat[id].values());
        print(state.task);

