import os
from pathlib import Path
import pickle
from time import process_time

import numpy as np
from tqdm import tqdm
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
from training_data import TrainingDataManager
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
import json
from skimage.io import imshow

data_folder = Path("H:\\Other computers\\My Computer\\")
fitness_folder = data_folder/'smb1Py'/'run_10_fitness_history';
dat:dict[int,dict[int,float]] = None;  # type: ignore

out_folder = Path("data");

start_gen = 1533;

# target = 1547;

process_all = False;

for f in os.listdir(fitness_folder):
    f = Path(f);
    if (os.path.exists(out_folder/f.with_suffix('.gz')) and not process_all):
        continue;
    num = int(str(f).split('_')[1]);
    if num < start_gen:
        continue;
    print(num);
    p = fitness_folder/f; 
    p = p.resolve();
    print(p);
    with open(p,'rb') as f:
        dat = pickle.load(f);

    TDM = TrainingDataManager[SegmentState]('smb1Py',10,data_folder=data_folder);

    game = tools.Control();
    state_dict = {c.LEVEL: Segment()}
    game.setup_states(state_dict, c.LEVEL)
    started = False;
    tile_scale = 4;
    view_distance = 6; #doesnt' really matter

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



    with open((out_folder/p.name).with_suffix('.gz'),'wb') as f:
        pickle.dump(training_data,f);

