from __future__ import annotations
from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
import os
import pickle
from typing import Any, Generic, Iterable, Iterator, MutableMapping, Protocol, TypeVar,Callable
from interrupt import DelayedKeyboardInterrupt
from filelock import FileLock
import shelve
from tqdm import tqdm

TD = TypeVar('TD');
class TrainingDataManager(Generic[TD]):

    def __init__(self,game_name,run,data_folder:os.PathLike="memories",ext="tdat",override_filepath:str|Path|os.Pathlike|None=None):
        self.data_file:Path = Path(override_filepath) if override_filepath else Path(data_folder)/game_name/(f"run-{run}.{ext}");
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
        
class Metadatum(Generic[TD]):
    def __init__(self,value:TD):
        self.value = value;

    def datum(self):
        return self.value;

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
    def iterator(scheduled_data:Iterator[Iterable[int]],startgen:int|None=None)->TDSource:
        return IteratorTDSource(scheduled_data,startgen=startgen);

    @staticmethod
    def generator(generator_func:Callable[[int],Iterable[TD]]):
        return GeneratorTDSource(generator_func);
        
class GeneratorTDSource(TDSource[TD],Generic[TD]):
    def __init__(self,gen): self.genr = gen;
    def __call__(self, gen: int) -> Iterable[TD]: 
        try: return self.genr(gen); 
        except TypeError: return self.genr();
    def empty(self):
        return False;

# DOESN'T WORK UNLESS USED IN ASYNCIO LOOP
# class AsyncStreamTDSource(TDSource[TD],Generic[TD]):
#     def __init__(self): 
#         self.event = Event(); 
#         self.data = None;
#         self.done = False;
#     async def __call__(self) -> Iterable[TD]: 
#         await self.event.wait();
#         self.event.clear();
#         return self.data;
#     def put_data(self,data:Iterable[TD]):
#         if self.event.is_set():
#             raise Exception("Error: Cannot add extra data to the stream before empty")
#         self.data = data;
#         self.event.set();
#     def empty(self):
#         return self.done;
#     def end_stream(self):
#         self.done = True;

class ConstantTDSource(TDSource[TD],Generic[TD]):
    def __init__(self,data:Iterable[TD]): self.data = data;
    def __call__(self, gen: int): return self.data;
    def empty(self): return False;

class IteratorTDSource(TDSource[TD],Generic[TD]):
    def __init__(self,scheduled_data:Iterator[Iterable[TD]],startgen:int|None=None): self.iterator = scheduled_data; self.start = startgen; self.finished = False;
    def __call__(self, gen: int) -> Iterable[TD]: 
        if self.start == gen: self.start = None;
        if self.start is None:
            try: return next(self.iterator)
            except StopIteration: self.finished = True; return [];
        else:
            return [];
    def empty(self) -> bool: return self.finished


class SourcedTDMixin(Generic[TD]):

    def __init__(self,game_name,run,*args,initial_sources:list[TDSource[TD]],data_folder:os.PathLike="memories",ext="tdat",override_filepath=None,**kwargs):
        super().__init__(game_name,run,*args,data_folder=data_folder,ext=ext,override_filepath=override_filepath,**kwargs);
        self.sources = {}
        self.add_source(*initial_sources);
        self.active_data:dict[int,TD]
        self.source_map = {};
        self._next_source_id = 0;

    def next_source_id(self):
        id = self._next_source_id;
        self._next_source_id += 1;
        return id;

    def get_datum_source(self,id:int):
        return self.source_map[id] if id in self.source_map else None;

    def add_source(self,*sources:TDSource[TD]):
        [self.sources.update({self.next_source_id:s}) for s in sources];

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

    def source_next_id(self,source:TDSource):
        id:int = self.next_id();
        self.source_map[id] = source;
        return id;

    @contextmanager
    def poll_data(self,generation:int):
        try:
            self.source_map = {};
            for source in self.sources:
                self.active_data.update([(self.source_next_id(source),datum) for datum in source(generation)]);
            yield self.active_data;
        finally:
            self.clear_data(save=True);


class ShelvedTDManager(ShelvedTDMixin[TD],TrainingDataManager[TD],Generic[TD]): pass;
class SourcedTDManager(SourcedTDMixin[TD],TrainingDataManager[TD],Generic[TD]): pass;
class SourcedShelvedTDManager(ShelvedTDMixin[TD],SourcedTDMixin[TD],TrainingDataManager[TD],Generic[TD]): pass;


class DataQueue(Generic[TD]):
    def __init__(self,queue_file:os.PathLike):
        self.queue_file = queue_file;

    def enqueue_data(self,data:Iterable[TD],):
        with FileLock(self.queue_file):
            with open(self.queue_file,'rb') as f:
                queue:list[Iterable[TD]] = pickle.load(f);
                queue.insert(0,data)
                
    
