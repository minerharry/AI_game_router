from __future__ import annotations
from lib2to3.pgen2.pgen import generate_grammar
from pathlib import Path
import os
import pickle
import queue
import sys
from typing import Generic, Iterable, MutableMapping, TypeVar,Callable
from interrupt import DelayedKeyboardInterrupt
from neat.reporting import BaseReporter
from filelock import FileLock
import shelve
from tqdm import tqdm

TD = TypeVar('TD');
class TrainingDataManager(Generic[TD],BaseReporter):

    def __init__(self,game_name,run,data_folder:os.PathLike="memories",ext="tdat",generation_func:Callable[[],Iterable[TD]]|None=None):
        self.data_file:Path = Path(data_folder)/game_name/(f"run-{run}.{ext}");
        self.generator = generation_func;
        if (os.path.exists(self.data_file)):
            self.load_data();
        else:
            self.create_blank();
            self.save_data();

    def create_blank(self):
        self.next_id:int = 0;
        self.active_data:dict[int,TD] = {};
        self._inactive_data:dict[int,TD] = {};

    def __len__(self):
        return len(self.active_data);

    def clear_data(self,save=True):
        for id,datum in self.active_data.items():
            self._inactive_data[id] = datum;
        self.active_data = {};
        if save: self.save_data();

    def add_data(self,data:Iterable[TD],save=True):
        ids_added = [];
        for datum in data:
            self.active_data[self.next_id] = datum;
            ids_added.append(self.next_id);
            self.next_id += 1;
        if save: self.save_data();
        return ids_added;

    def set_data(self,data:Iterable[TD],save=True):
        self.clear_data(save=False);
        return self.add_data(data,save=save);

    def load_data(self):
        with DelayedKeyboardInterrupt():
            with open(self.data_file,'rb') as f:
                ob = pickle.load(f);
                self.next_id = ob['next_id'];
                self.active_data = ob['active_data'];
                try:
                    self._inactive_data = ob['inactive_data'];
                except KeyError:
                    if ('shelf' in ob):
                        raise Exception("Unable to find inactive data in save object. A shelf filename was found - did you mean to use ShelvedTDManager instead?");


    def save_data(self):
        out = {'next_id':self.next_id, 'active_data':self.active_data, 'inactive_data':self._inactive_data};
        pickle.dumps(out); #test pickle for breaking
        with DelayedKeyboardInterrupt():
            with open(self.data_file,'wb') as f:
                try:
                    pickle.dump(out,f);
                except TypeError as e:
                    raise e;

    def get_data_by_id(self,id):
        if id in self.active_data:
            return self.active_data[id];
        elif id in self._inactive_data:
            return self._inactive_data[id];
        else:
            raise IndexError(f"Id {id} not in active or inactive data");

    def end_generation(self, config, population, species_set):
        if self.generator:
            self.set_data(list(self.generator()));

    def __getitem__(self,idx):
        return self.get_data_by_id(idx)
            
class ShelvedTDManager(TrainingDataManager[TD]):
    def create_blank(self):
        self.next_id:int = 0;
        self.active_data:dict[int,TD] = {};
        self._shelf_file = self.data_file.with_suffix(".tdac");
        if not os.path.exists(self._shelf_file.parent):
            os.makedirs(self._shelf_file.parent);
        self._shelf = shelve.DbfilenameShelf[TD](str(self._shelf_file));

    def clear_data(self, save=True):
        for id,datum in self.active_data.items():
            self._shelf[str(id)] = datum;
        self.active_data = {};
        if save: self.save_data();

    def get_data_by_id(self, id):
        if id in self.active_data:
            return self.active_data[id];
        elif str(id) in self._shelf:
            return self._shelf[str(id)];
        else:
            raise IndexError(f"Id {id} not in active or inactive data");

    def save_data(self):
        out = {'next_id':self.next_id, 'active_data':self.active_data,'shelf':self._shelf_file};
        pickle.dumps(out);
        with DelayedKeyboardInterrupt():
            with open(self.data_file,'wb') as f:
                try:
                    pickle.dump(out,f);
                except TypeError as e:
                    raise e;
        self._shelf.sync();

    def load_data(self):
        #shelf is already always loaded
    
        with DelayedKeyboardInterrupt():
            with open(self.data_file,'rb') as f:
                ob = pickle.load(f);
                self.next_id = ob['next_id'];
                self.active_data = ob['active_data'];
                self._shelf_file = ob['shelf'] if 'shelf' in ob else self.data_file.with_suffix(".tdac");
                temp_inactive = ob['inactive_data'] if 'inactive_data' in ob else None; #for conversion from base tdmanager compatiblity
        self._shelf = shelve.DbfilenameShelf[TD](str(self._shelf_file));
        if temp_inactive:
            print("previous data entry found, loading into shelf...");
            for k,v in tqdm(temp_inactive.items()):
                self._shelf[str(k)] = v;
            self.save_data();


        

    


class DataQueue(Generic[TD]):
    def __init__(self,queue_file:os.PathLike):
        self.queue_file = queue_file;

    def enqueue_data(self,data:Iterable[TD],):
        with FileLock(self.queue_file):
            with open(self.queue_file,'rb') as f:
                queue:list[Iterable[TD]] = pickle.load(f);
                queue.insert(0,data)
                
    
