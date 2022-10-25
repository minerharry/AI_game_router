"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import random
import time
from interrupt import DelayedKeyboardInterrupt

try:
    import dill as pickle
except ImportError:
    try:
        import cPickle as pickle # pylint: disable=import-error
    except ImportError:
        import pickle # pylint: disable=import-error


from neat.population import Population
from neat.reporting import BaseReporter
from neat.six_util import itervalues


class Checkpointer(BaseReporter):
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """
    def __init__(self, generation_interval=100, time_interval_seconds=300,
                 filename_prefix='neat-checkpoint-',file_extension=".gz"):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self.generation_interval = generation_interval
        self.time_interval_seconds = time_interval_seconds
        self.filename_prefix = filename_prefix
        self.file_ext = file_extension;

        self.current_generation = None
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()

    def start_generation(self, generation):
        self.current_generation = generation

    def end_generation(self, config, population, species_set):
        checkpoint_due = False

        if self.time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self.time_interval_seconds:
                checkpoint_due = True

        if (checkpoint_due is False) and (self.generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            if dg >= self.generation_interval:
                checkpoint_due = True

        if checkpoint_due:
            self.save_checkpoint(config, population, species_set, self.current_generation)
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    def save_checkpoint(self, config, population, species_set, generation):
        """ Save the current simulation state. """
        filename = '{0}{1}{2}'.format(self.filename_prefix,generation,self.file_ext)
        print("Saving checkpoint to {0}".format(filename))
        
        with DelayedKeyboardInterrupt():
            with gzip.open(filename, 'w', compresslevel=5) as f:
                data = (generation, config, population, species_set, random.getstate())
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename,config_transfer=None)->Population:
        """Resumes the simulation from a previous saved point."""
        with DelayedKeyboardInterrupt():
            with gzip.open(filename) as f:
                generation, config, population, species_set, rndstate = pickle.load(f)
                random.setstate(rndstate)
                if config_transfer: #transfer configuration and update population and species_set from config
                    print(f'config transfer: {config_transfer}');
                    newConfig, inputMap, outputMap = config_transfer;
                    if inputMap:
                        for genome in itervalues(population):
                            genome.remap_inputs(inputMap,config.genome_config,newConfig.genome_config);
                    if outputMap:
                        for genome in itervalues(population):
                            genome.remap_outputs(outputMap,config.genome_config,newConfig.genome_config);
                    config = newConfig;

                pop = Population(config, (population, species_set, generation+1));
                species_set.reporters = pop.reporters;
                return pop;