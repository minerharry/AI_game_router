from __future__ import annotations
from abc import abstractmethod
from contextlib import contextmanager
from lib2to3.pgen2.pgen import generate_grammar
from pathlib import Path
import os
import pickle
import queue
import sys
from typing import Any, Generic, Iterable, MutableMapping, Protocol, TypeVar,Callable
from interrupt import DelayedKeyboardInterrupt
from neat.reporting import BaseReporter
from filelock import FileLock
import shelve
from tqdm import tqdm


TD = TypeVar('TD');
class TrainingDataManager(Generic[TD]):

    def __init__(self,game_name,run,data_folder:os.PathLike="memories",ext="tdat",override_filepath:os.Pathlike=None):
        self.data_file:Path = Path(override_filepath) or Path(data_folder)/game_name/(f"run-{run}.{ext}");
        if (os.path.exists(self.data_file)):
            self.load_data();
        else:
            self.create_blank();
            self.save_data();

    def __call__(self, gen:int,*args: Any, **kwargs: Any):
        return self.poll_data(gen,*args,**kwargs);

    def create_blank(self):
        self._next_id:int = 0;
        self.active_data:dict[int,TD] = {};
        self._inactive_data:dict[int,TD] = {};


    def clear_data(self,save=True):
        for id,datum in self.active_data.items():
            self._inactive_data[id] = datum;
        self.active_data = {};
        if save: self.save_data();

    def next_id(self):
        out = self._next_id;
        self._next_id += 1;
        return out;

    def add_data(self,data:Iterable[TD],save=True):
        ids_added = [];
        for datum in data:
            self.active_data[self.next_id()] = datum;
        if save: self.save_data();
        return ids_added;

    def set_data(self,data:Iterable[TD],save=True):
        self.clear_data(save=False);
        return self.add_data(data,save=save);

    @contextmanager
    def poll_data(self,gen:int):
        yield self;

    def load_data(self):
        with DelayedKeyboardInterrupt():
            with open(self.data_file,'rb') as f:
                ob = pickle.load(f);
                self._next_id = ob['next_id'];
                self.active_data = ob['active_data'];
                try:
                    self._inactive_data = ob['inactive_data'];
                except KeyError:
                    if ('shelf' in ob):
                        raise Exception("Unable to find inactive data in save object. A shelf filename was found - did you mean to use ShelvedTDManager instead?");


    def save_data(self):
        out = {'next_id':self._next_id, 'active_data':self.active_data, 'inactive_data':self._inactive_data};
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

    def __getitem__(self,idx):
        return self.get_data_by_id(idx)
        
    
    def __len__(self):
        return len(self.active_data);

    def __iter__(self):
        return iter(self.active_data);

    def items(self):
        return self.active_data.items();
        
            
class ShelvedTDMixin(Generic[TD]):
    def create_blank(self):
        super().create_blank();
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
        out = {'next_id':self._next_id, 'active_data':self.active_data,'shelf':self._shelf_file};
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
                self._next_id = ob['next_id'];
                self.active_data = ob['active_data'];
                self._shelf_file = ob['shelf'] if 'shelf' in ob else self.data_file.with_suffix(".tdac");
                temp_inactive = ob['inactive_data'] if 'inactive_data' in ob else None; #for conversion from base tdmanager compatiblity
        self._shelf = shelve.DbfilenameShelf[TD](str(self._shelf_file));
        if temp_inactive:
            print("previous data entry found, loading into shelf...");
            for k,v in tqdm(temp_inactive.items()):
                self._shelf[str(k)] = v;
            self.save_data();


# TDSource = Callable[[int],Iterable[TD]]
class TDSource(Protocol[TD]):
    @abstractmethod
    def __call__(self, gen:int) -> Iterable[TD]: pass;

    @abstractmethod
    def empty(self) -> bool: 
        """When TDSource.empty() returns true, a SourcedTDMixin implementer will remove this source from its list of active sources. This allows for schedululing sources a finite time into the future without polling from many empty sources."""
        return False;
        
    @staticmethod
    def constant(data:Iterable[TD])->TDSource:
        return ConstantTDSource(data);

    @staticmethod
    def schedule(scheduled_data:Iterable[Iterable[int]],startgen:int|None=None)->TDSource:
        return ScheduledTDSource(scheduled_data,startgen=startgen);

    @staticmethod
    def generator(generator_func:Callable[[int],Iterable[TD]]):
        return GeneratorTDSource(generator_func);
        
class GeneratorTDSource(TDSource[TD],Generic[TD]):
    def __init__(self,gen): self.genr = gen;
    def __call__(self, gen: int) -> Iterable[TD]: 
        try: return self.genr(gen); 
        except TypeError: return self.genr();

class ConstantTDSource(TDSource[TD],Generic[TD]):
    def __init__(self,data:Iterable[TD]): self.data = data;
    def __call__(self, gen: int): return self.data;
    def empty(self): return False;

class ScheduledTDSource(TDSource[TD],Generic[TD]):
    def __init__(self,scheduled_data:Iterable[Iterable[int]],startgen:int|None=None): self.iterator = iter(scheduled_data); self.start = startgen; self.finished = False;
    def __call__(self, gen: int) -> Iterable[TD]: 
        if self.start == gen: self.start = None;
        if self.start is None:
            try: return next(self.iterator)
            except StopIteration: self.finished = True; return [];
    def empty(self) -> bool: return self.finished


class SourcedTDMixin(Generic[TD]):

    def __init__(self,sources:list[TDSource[TD]],game_name,run,*args,data_folder:os.PathLike="memories",ext="tdat",override_filepath=None,**kwargs):
        super().__init__(game_name,run,*args,data_folder=data_folder,ext=ext,override_filepath=override_filepath,**kwargs);
        self.sources = sources
        self.active_data:dict[int,TD]

    def add_source(self,*sources:TDSource[TD]):
        [self.sources.append(s) for s in sources];

    def remove_source(self,source):
        self.sources.remove(source);

    def get_sources(self):
        return self.sources;
    
    def clear_sources(self):
        self.sources = [];

    def add_data(self,*args,**kwargs):
        raise NotImplementedError("adding data directly not supported for source-based training data manager; add a constant source instead")
    
    def set_data(self,*args,**kwargs):
        raise NotImplementedError("setting data directly not supported for source-based training data manager; add a constant source instead")

    @contextmanager
    def poll_data(self):
        try:
            for source in self.sources:
                self.active_data.update([self.next_id(),datum for datum in source(generation)]);
            yield self.active_data;
        finally:
            self.clear_data(save=True);


class ShelvedTDManager(TrainingDataManager[TD],ShelvedTDMixin[TD],Generic[TD]): pass;
class SourcedTDManager(TrainingDataManager[TD],SourcedTDMixin[TD],Generic[TD]): pass;
class SourcedShelvedTDManager(TrainingDataManager[TD],SourcedTDMixin[TD],ShelvedTDMixin[TD],Generic[TD]): pass;


class DataQueue(Generic[TD]):
    def __init__(self,queue_file:os.PathLike):
        self.queue_file = queue_file;

    def enqueue_data(self,data:Iterable[TD],):
        with FileLock(self.queue_file):
            with open(self.queue_file,'rb') as f:
                queue:list[Iterable[TD]] = pickle.load(f);
                queue.insert(0,data)
                
    
