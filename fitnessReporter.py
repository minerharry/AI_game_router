from pathlib import Path
from typing import DefaultDict, NamedTuple
from interrupt import DelayedKeyboardInterrupt
from neat.reporting import BaseReporter
import os
try:
    import cPickle as pickle
except:
    import pickle


class GenomeFitness(NamedTuple):
    fitness:float|list[float]
    genome_id:int = -1;


class FitnessCheckpoint:
    def __init__(self,data:dict[int,dict[int,float|list[float]]|list[float|list[float]]]):
        self.data:dict[int, list[GenomeFitness]] = {};
        for tdat_id, tdat_data in data:
            res:list[GenomeFitness];
            if isinstance(tdat_data,dict):
                ## genome_id mapped fitness data
                res = [GenomeFitness(data,id) for id,data in tdat_data.items()];
            else:
                res = [GenomeFitness(data,-1) for data in tdat_data];
            self.data[tdat_id] = res;

    def save_checkpoint(self,path:str|Path,override_ext=True):
        path = Path(path);
        if (override_ext):
            path = path.with_suffix(".fit");
        with DelayedKeyboardInterrupt():
            with open(path,'wb') as f:
                pickle.dump(self,f);

    @classmethod
    def load_checkpoint(cls,path:str):
        with DelayedKeyboardInterrupt():
            with open(path,'rb') as f:
                out = pickle.load(f);
        return out;

    # @classmethod
    # def from_task_fitness_data(self,task_data:list[]):



class FitnessReporter(BaseReporter):

    def __init__(self,gameName,run_name):
        os.makedirs(f"memories/{gameName}/{run_name}_fitness_history",exist_ok=True);
        self.gameName = gameName;
        self.run_name = run_name;

    def save_data(self,fitness_data):
        path = f"memories/{self.gameName}/{self.run_name}_fitness_history/gen_{self.generation}";
        checkpoint = FitnessCheckpoint(fitness_data);
        checkpoint.save_checkpoint(path);
        
    def start_generation(self,generation):
        self.generation = generation

        


    
