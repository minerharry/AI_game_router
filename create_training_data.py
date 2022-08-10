from pathlib import Path
import pickle

import numpy as np
from tqdm import tqdm
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
from training_data import TrainingDataManager
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
import json

fitness_folder = Path("transfer")/'smb1Py';
dat:dict[int,dict[int,float]] = None;
p = fitness_folder/'run-10'/'gen_1536';
p = p.resolve();
print(p);
with open(p,'rb') as f:
    dat = pickle.load(f);

TDM = TrainingDataManager[SegmentState]('smb1Py',10,data_folder='transfer');

game = tools.Control();
state_dict = {c.LEVEL: Segment()}
game.setup_states(state_dict, c.LEVEL)
started = False;
tile_scale = 8;
view_distance = 6;

training_data = []

print(list(dat.keys())[:20])

for id,fitnesses in tqdm(dat.items()):
    state = TDM[id];
    # state.static_data[c.MAP_MAPS][0][c.MAP_START][1] -= 12;
    if not started:
        game.state.startup(0,{c.LEVEL_NUM:1},initial_state=state);
        started = True;
    else:
        game.load_segment(state);
    # sstate:Segment = game.state;
    # print(sstate.player.rect.center);
    # sstate.player.rect.move(0,10);
    # print(sstate.player.rect.center);
    gdat = game.get_game_data(view_distance,tile_scale);
    mdat = game.get_map_data(tile_scale);
    training_data.append(((gdat,mdat),max(fitnesses.values())));

out_folder = Path("data");

with open((out_folder/p.name).with_suffix('.gz'),'wb') as f:
    pickle.dump(training_data,f);

