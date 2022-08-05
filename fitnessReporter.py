from interrupt import DelayedKeyboardInterrupt
from neat.reporting import BaseReporter
import os
try:
    import cPickle as pickle
except:
    import pickle


class FitnessReporter(BaseReporter):

    def __init__(self,gameName,run_name):
        os.makedirs(f"memories\\{gameName}\\{run_name}_fitness_history",exist_ok=True);
        self.gameName = gameName;
        self.run_name = run_name;

    def save_data(self,fitness_data):
        print(list(fitness_data));
        with DelayedKeyboardInterrupt:
            with open(f"memories\\{self.gameName}\\{self.run_name}_fitness_history\\gen_{self.generation}",'wb') as f:
                pickle.dump(fitness_data,f)

    def start_generation(self,generation):
        self.generation = generation

        


    
