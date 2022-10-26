from copy import copy
import os
from pathlib import Path
import pickle
from time import process_time
import statistics
import numpy as np
from tqdm import tqdm
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState
from training_data import TrainingDataManager
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as c
import json
from skimage.io import imshow
import copy

data_folders = [Path("H:/Other computers/My Computer/"),Path("memories")][:1];
start_gens = [1583,1603][:1];
runs = [10,'play_test'][:1];
out_folder = Path("data");
for n,(data_folder,start_gen,run) in tqdm(enumerate(zip(data_folders,start_gens,runs)),total=len(runs)):

    fitness_folder = data_folder/'smb1Py'/f'run_10_fitness_history';
    dat:dict[int,dict[int,float|dict[int,float]]|list[float|dict[int,float]]] = None;  # type: ignore
    #datum_id : {(optional) genome_id : fitness | level_segment_fitnesses}
    process_all = True;

    to_process = [];

    for f in os.listdir(fitness_folder):
        f = Path(f);
        if (os.path.exists((out_folder/(f'{n}_{f}')).with_suffix('.gz')) and not process_all):
            continue;
        num = int(str(f).split('_')[1]);
        if num < start_gen:
            continue;
        to_process.append(f);
        
    for f in tqdm(to_process):
        f = Path(f);


        p = fitness_folder/f; 
        p = p.resolve();

        with open(p,'rb') as f:
            dat = pickle.load(f);

        TDM = TrainingDataManager[SegmentState]('smb1Py',run,data_folder=data_folder);

        game = tools.Control();
        state_dict = {c.LEVEL: Segment()}
        game.setup_states(state_dict, c.LEVEL)
        started = False;
        tile_scale = 4;
        view_distance = 6; #doesn't really matter

        training_data = []


        for id,task_fitnesses in tqdm(dat.items()):
            if isinstance(task_fitnesses,dict):
                task_fitnesses = np.array(task_fitnesses.values());
            else:
                task_fitnesses = np.array(task_fitnesses);
            
            if (len(task_fitnesses.shape) == 1):
                np.expand_dims(task_fitnesses,0);
            
            state = TDM[id];
            if not started:
                game.state.startup(0,{c.LEVEL_NUM:1},initial_state=state);
                started = True;
            else:
                game.load_segment(state);


            for flist in task_fitnesses:
                gdat = dict(game.get_game_data(view_distance,tile_scale));
                mdat = dict(game.get_map_data(tile_scale));
                
                fitness_data = {
                    'mean':np.average(flist),
                    'median':np.median(flist),
                    'quantiles':[np.quantile(flist,0.25),np.quantile(flist,0.75)],
                    'max':np.max(flist),
                    'min':np.min(flist),
                }

                training_data.append(((gdat,mdat),fitness_data));

                s:Segment = game.state;
                s.increment_task_path(move_player=True);



        with open((out_folder/(f'{n}_{p.name}')).with_suffix('.gz'),'wb') as f:
            pickle.dump(training_data,f);

