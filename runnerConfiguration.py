from __future__ import annotations
from typing import Callable, TypeVar
import numpy as np

from training_data import TrainingDataManager

class RunnerConfig:

    def __init__(self,
            gameFitnessFunction:Callable[[dict],float],
            gameRunningFunction:Callable[[dict],float],
            training_data:TrainingDataManager|None=None,
            logging=False,logPath='',
            recurrent=False,
            trial_fitness_aggregation='average',custom_fitness_aggregation:None|Callable[[list[float]],float]=None,
            time_step=0.05,
            num_trials=10,
            parallel=False,parallel_processes=4,
            pool_type="multiprocessing",queue_type="multiprocessing",
            returnData:list[str|IOData]=[],
            gameName='game',
            num_generations:int|None=300,
            fitness_collection_type='final',
            serialize_reporters=False,
            **kwargs):
        self.logging = logging;
        self.logPath = logPath;
        self.generations = num_generations;
        self.recurrent = recurrent;
        self.gameName = gameName;
        self.parallel = parallel;
        self.parallel_processes = 4;
        self.pool_type=pool_type;
        self.queue_type=queue_type;
        if "ray" in [self.queue_type,self.pool_type]:
            print("ray detected, activating...");
            import ray
            ray.init(ignore_reinit_error=True);

        self.serialize_reporters = serialize_reporters;

        self.time_step = time_step;
        self.numTrials = num_trials;
        self.fitnessFromGameData = gameFitnessFunction;
        self.gameStillRunning = gameRunningFunction;
        self.fitness_collection_type = fitness_collection_type;
        self.returnData:list[str|IOData] = returnData;
        self.training_data = training_data;
        if (trial_fitness_aggregation == 'custom'):
            self.customFitnessFunction = custom_fitness_aggregation;
        else:
            self.customFitnessFunction = None;
        self.trialFitnessAggregation = trial_fitness_aggregation;
        self.reporters = [];

        for name,arg in kwargs.items():
            setattr(self,name,arg);

    def __getstate__(self): #pickling
        state = self.__dict__.copy();
        if not self.serialize_reporters:
            state["reporters"] = [];
        return state;

    @property
    def fitnessFromArray(self)->Callable[[list[float]],float]:
        if self.customFitnessFunction is not None:
            return self.customFitnessFunction;
        elif self.trialFitnessAggregation == 'average':
            return lambda array : sum(array) / len(array);
        elif self.trialFitnessAggregation == 'sum':
            return lambda array : sum(array);
        elif self.trialFitnessAggregation == 'max':
            return lambda array : max(array);
        elif self.trialFitnessAggregation == 'min':
            return lambda array: min(array);
        else:
            raise Exception(f'error: fitness aggregation function {self.trialFitnessAggregation} not defined');

    #named input data, flattened
    def flattened_return_data(self):
        result = [];
        for datum in self.returnData:
            if (isinstance(datum,IOData)):
                [result.append(x) for x in datum.getSplitData()];
            else:
                result.append(datum);
#        print(result);
        return result;

    #named input data, shaped
    def return_data_shape(self):
        result = [];
        for datum in self.returnData:
            if (isinstance(datum,IOData)):
                result.append(datum.toNamedArray());
                print('name array of {0}: {1}'.format(datum,datum.toNamedArray()));
            else:
                result.append(datum);
        return result;          
    
    def get_input_transfer(self,prevData:list):
        newData = self.flattened_return_data();
        result = []
        for datum in prevData:
            if (isinstance(datum,IOData)):
                [result.append(x) for x in datum.getSplitData()];
            else:
                result.append(datum);
        prevData = result;
        map = {};
        unused_keys = list(range(len(newData)));
        for i in range(len(prevData)):
            name = prevData[i];
            if name in newData:
                index = newData.index(name);
                unused_keys.remove(index);
                map[-i-1] = -index-1;
            else:
                map[-i-1] = None
        map[None] = [-key-1 for key in unused_keys]; #technically it doesn't actually add nodes when negative, but here for consistency's sake
        print(map[None]);
        return map;


def get_array_cell_names(array_size):
    if (len(array_size) == 1):
        return [str(i) for i in range(array_size[0])];
    return [str(i) + '-' + j for i in range(array_size[0]) for j in get_array_cell_names(array_size[1:])];

def sum_arrays(array:list,depth=1):
    if depth == 1:
        return array;
    if depth == 2:
        result = [];
        for a in array:
            result += a;
        return result;
    result = [];
    for a in array:
        result += sum_arrays(a,depth-1);
    return result;

class IOData:
    convertable_types = [list];
    arrays = ['array','ndarray'];
    
    def __init__(self,name,data_type,array_size=None):
        self.data_type = data_type;
        self.name = name;
        self.array_size = array_size;
        
    #returned a flattened list of all return data
    def getSplitData(self):
        if (self.data_type == 'float'):
            return [self.name];
        if (self.data_type in self.arrays):
            return [self.name + ' ' + x for x in get_array_cell_names(self.array_size)];

    def toNamedArray(self):
        return self.toNameArray(self.name,self.array_size,' ');

    def toNameArray(self,name,shape,splitterChar):
        result = [];
        if (len(shape) == 1):
            for i in range(shape[0]):
                result.append(str(name) + splitterChar + str(i));
        else:
            for i in range(shape[0]):
                result.append(self.toNameArray(str(name) + splitterChar + str(i),shape[1:],'-'));
        print(result);
        return (name,result);

    def extract_data(self,datum)->list[float]:
        if (self.data_type == 'float'):
            return datum;
        if (self.data_type == 'array'):
            return sum_arrays(datum,len(self.array_size));





