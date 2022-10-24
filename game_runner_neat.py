import re
from typing import Any, Callable, Generic, Type
from cv2 import trace
import pandas as pd
import ray
from ray.util.queue import _QueueActor
from tqdm import tqdm
from baseGame import EvalGame, RunGame
import neat
import tracemalloc
import traceback
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from play_level import TaskFitnessReporter
from runnerConfiguration import RunnerConfig
import os.path
import os
import visualize
import sys
import random
import functools
from fitnessReporter import FitnessReporter
from datetime import datetime
import istarmap
import linecache
import multiprocessing
from logReporting import LoggingReporter
from renderer import Renderer as RendererReporter
from videofig import videofig as vidfig
from neat.six_util import iteritems, itervalues
try:
    from viztracer import log_sparse
except:
    pass;
try:
    from pympler import tracker,asizeof,summary
except:
    tracker,asizeof = None,None;

import ray
from ray.util.queue import Queue
from modified_pool import Pool


#requires get_genome_frame.images to be set before call
def get_genome_frame(f,axes):
    images = get_genome_frame.images;
    if not get_genome_frame.initialized:
        get_genome_frame.im = axes.imshow(images[f],animated=True);
        get_genome_frame.initialized = True;
    else:
        get_genome_frame.im.set_array(images[f]);
            
class GameRunner:

    #if using default version, create basic runner and specify game to run
    def __init__(self,game:EvalGame,runnerConfig:RunnerConfig):
        self.game = game;
        self.runConfig = runnerConfig;
        self.generation:int = None;

    def continue_run(self,run_name,render=False,manual_generation=None,manual_config_override=None,single_gen=False):
        checkpoint_folder = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_');
        if manual_generation is None:
            files = os.listdir(checkpoint_folder);
            maxGen = -1;
            for file in files:
                m = re.match("run-checkpoint-([0-9]+)",file);
                if m:
                    gen = int(m.group(1));
                    if (gen>maxGen):
                        maxGen = gen;
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '\\run-checkpoint-' + str(maxGen) + ".gz",config_transfer=manual_config_override);
        else:
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '\\run-checkpoint-' + str(manual_generation) + ".gz",config_transfer=manual_config_override);

        return self.run(pop.config,run_name,render=render,pop=pop,single_gen=single_gen);

    def replay_generation(self,generation,run_name,render=False,genome_config_edits=None):
        checkpoint_folder = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_');
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '\\run-checkpoint-' + str(generation) + '.gz');

        config = pop.config;

        if (genome_config_edits is not None):
            for k,v in genome_config_edits:
                if hasattr(config.genome_config,k):
                    setattr(config.genome_config,k,v);

        return self.run(config,run_name,render=render,pop=pop,single_gen=True);

    def run(self,config,run_name,render=False,pop=None,single_gen=False,force_fitness=False):
        self.run_name = run_name.replace(' ','_');
        if (pop is None):
            pop = neat.Population(config);
            continuing = False;
        else:
            continuing = True;
        stats = neat.StatisticsReporter();
        if (self.runConfig.logging):
            logReporter = LoggingReporter(self.runConfig.logPath,True);
            pop.add_reporter(logReporter);
        pop.add_reporter(stats);
        pop.add_reporter(neat.StdOutReporter(True));

        if not single_gen:
            os.makedirs("checkpoints\\games\\"+self.runConfig.gameName.replace(' ','_')+f'\\{self.run_name}',exist_ok=True);            
            pop.add_reporter(neat.Checkpointer(1,filename_prefix='checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+self.run_name+'\\run-checkpoint-'));

        if (render):
            pop.add_reporter(RendererReporter(self));
        if (hasattr(self.runConfig,'reporters') and self.runConfig.reporters != None):
            for reporter in self.runConfig.reporters:
                pop.add_reporter(reporter);
        if (continuing):
            pop.reporters.checkpoint_restored(pop.generation);
            # pop.complete_generation();
        
        if self.runConfig.parallel and not hasattr(self,'pool'):
            idQueue = Queue();
            [idQueue.put(i) for i in range(self.runConfig.parallel_processes)];
            self.pool = GenomeExecutorPool(self.runConfig.parallel_processes, initargs=(idQueue,self.game,config));
            

        if not single_gen or force_fitness:
            self.fitness_reporter = FitnessReporter(self.runConfig.gameName,self.run_name);
            pop.add_reporter(self.fitness_reporter);

        self.generation = pop.generation;
        
        winner = pop.run(self.eval_genomes,self.runConfig.generations if not single_gen else 1);

        return winner;

    def check_output_connections(self,generation,run_name,target_output,render=False):
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        connected = [];
        for g in itervalues(pop.population):
            for connection in g.connections:
                if (connection[1] == target_output):
                    connected.append(g);
                    break;
        [print (connectedGenome.key) for connectedGenome in connected];

    def render_worst_genome(self,generation,config,run_name,net=False):
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        worst = None
        for g in itervalues(pop.population):
            if worst is None or g.fitness < worst.fitness:
                worst = g
        if worst is not None:
            self.render_genome_by_id(worst.key,generation,config,run_name,net=net);
        else:
            raise Exception("no genomes to eval");

    def render_genome_by_id(self,genomeId,generation,config,run_name,net=False):
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        genome = None;
        for g in itervalues(pop.population):
            if g.key == genomeId:
                genome = g;
                break;
        self.render_genome(genome,config,net=net);
                    
    def render_custom_genome_object(self,obj,config,net=False):
        self.render_genome(obj,config,net=net)

    def replay_best(self,generation,config,run_name,net=False,randomReRoll=False,number=1):
        if number < 1:
            raise Exception("must replay at least one genome");
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        #self.eval_genomes(list(iteritems(pop.population)),config);
        if (randomReRoll):
            random.seed();
        sort = sorted(pop.population.items(),key=lambda x: x[0]);
        for _,g in sort[:number]:
            self.render_genome(g,config,net=net);

    def run_top_genomes(self,generation,config,run_name,number,doFitness=False,randomReRoll=False):
        checkpoint_folder = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_');
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '\\run-checkpoint-' + str(generation) + '.gz');

        config = pop.config;

        if self.runConfig.parallel:
            idQueue = Queue();
            [idQueue.put(i) for i in range(self.runConfig.parallel_processes)];
            self.pool = GenomeExecutorPool(self.runConfig.parallel_processes, initargs=(idQueue,self.game,config));
            
        self.run_name = run_name.replace(' ','_');
        if doFitness:
            self.fitness_reporter = FitnessReporter(self.runConfig.gameName,self.run_name + f"_top_{number}");
            self.fitness_reporter.start_generation(generation);

        if (randomReRoll):
            random.seed();

        sort = sorted(pop.population.items(),key=lambda x: x[0]);

        self.eval_genomes(sort[:number],config);
        

    def render_genome(self,genome,config,net=False):
        if (net):
            flattened_data = self.runConfig.flattened_return_data();
            shaped_data = self.runConfig.return_data_shape();
            visualize.draw_net(config,genome,view=True,node_names=dict([(-i-1,flattened_data[i]) for i in range(len(flattened_data))]),nodes_shape=shaped_data);
        if self.runConfig.training_data is None:
            if (self.runConfig.recurrent):  
                self.render_genome_recurrent(genome,config,net=net);
            else:
                self.render_genome_feedforward(genome,config,net=net);
        else:
            
            for datum in self.runConfig.training_data.active_data.values():
                if (self.runConfig.recurrent):  
                    self.render_genome_recurrent(genome,config,net=False,training_datum = datum);
                else:
                    self.render_genome_feedforward(genome,config,net=False,training_datum = datum);



    #render a genome with the game as a recurrent neural net
    def render_genome_recurrent(self, genome, config,net=False):
        runnerConfig = self.runConfig;
        time_const = runnerConfig.time_step;

        if (net):
            flattened_data = runnerConfig.flattened_return_data();
            shaped_data = runnerConfig.return_data_shape();
            visualize.draw_net(config,genome,view=True,node_names=dict([(-i-1,flattened_data[i]) for i in range(len(flattened_data))]),nodes_shape=shaped_data);
            
        if (runnerConfig.parallel and False):
            return;
            #TODO: implement parallel game processing
        else:
            net = neat.ctrnn.CTRNN.create(genome,config,time_const);
            runningGame = self.game.start(runnerConfig);
            images = [];
            #get the current data from the running game, as specified by the runnerConfig
            gameData = runningGame.getData();
            while (runningGame.isRunning(useCache=True)):
                #get the current inputs from the running game, as specified by the runnerConfig

                gameInput = net.advance(gameData, time_const, time_const);

                images.append(runningGame.tickRenderInput(gameInput));
                gameData = runningGame.getData();

                        
            runningGame.close();
            get_genome_frame.images = images;
            get_genome_frame.initialized = False;
            vidfig(len(images),get_genome_frame,play_fps=runnerConfig.playback_fps);

        

    #render a genome with the game as a feedforward neural net
    def render_genome_feedforward(self, genome, config,net=False,training_datum=None):
        runnerConfig = self.runConfig;
        if (net):
            flattened_data = runnerConfig.flattened_return_data();
            shaped_data = runnerConfig.return_data_shape();
            visualize.draw_net(config,genome,view=True,node_names=dict([(-i-1,flattened_data[i]) for i in range(len(flattened_data))]),nodes_shape=shaped_data);
            
            
        if (runnerConfig.parallel and False):
            return;
            #TODO: implement parallel game processing
        else:
            
            net = neat.nn.FeedForwardNetwork.create(genome,config);
            runningGame = self.game.start(runnerConfig,training_datum = training_datum);
            images = [];
            fitness = 0;
            if 'delta' in runnerConfig.fitness_collection_type:
                fitness -= runningGame.getFitnessScore();

            max_fitness = 0;
            if 'max' in runnerConfig.fitness_collection_type:
                max_fitness = runningGame.getFitnessScore();
            #get the current inputs from the running game, as specified by the runnerConfig
            gameData = runningGame.getData();
            while (runningGame.isRunning(useCache=True)):
        
                gameInput = net.activate(gameData);

                if (self.runConfig.external_render):
                    images.append(runningGame.tickRenderInput(gameInput));
                else:
                    runningGame.tickRenderInput(gameInput);


                if ('continuous' in runnerConfig.fitness_collection_type):
                    fitness += runningGame.getFitnessScore();
                elif ('max' in runnerConfig.fitness_collection_type):
                    max_fitness = max(max_fitness,runningGame.getFitnessScore());

                gameData = runningGame.getData();

            if 'max' in runnerConfig.fitness_collection_type:
                fitness += max_fitness;
            elif 'continuous' not in runnerConfig.fitness_collection_type: #prevent double counting
                fitness += runningGame.getFitnessScore();

            print('final genome fitness: ' + str(fitness));

                        
            runningGame.close();
            if (self.runConfig.external_render):
                get_genome_frame.images = images;
                get_genome_frame.initialized = False;
                vidfig(len(images),get_genome_frame,play_fps=runnerConfig.playback_fps);


    def eval_genomes(self,genomes,config):
        if (self.runConfig.recurrent):
            self.eval_genomes_recurrent(genomes,config);
        else:
            self.eval_genomes_feedforward(genomes,config);
        if self.generation is not None:
            self.generation += 1;
    

    #evaluate a population with the game as a recurrent neural net
    def eval_genomes_recurrent(self, genomes, config):
        runnerConfig = self.runConfig;
        time_const = runnerConfig.time_step;
        if (runnerConfig.parallel):
            return;
            #TODO: implement parallel game processing
        else:
            for genome_id, genome in genomes:
                net = neat.ctrnn.CTRNN.create(genome,config,time_const);
                fitnesses = [];
                for trial in range(runnerConfig.numTrials):
                    
                    runningGame = self.game.start(runnerConfig);
                    fitness = 0;
                    
                    #get the current inputs from the running game, as specified by the runnerConfig
                    gameData = runningGame.getData();
                    while (runningGame.isRunning(useCache=True)):

                        gameInput = net.advance(gameData, time_const, time_const);
                        
                        runningGame.tickInput(gameInput);
                        if (runnerConfig.fitness_collection_type != None and runnerConfig.fitness_collection_type == 'continuous'):
                            fitness += runningGame.getFitnessScore();

                        gameData = runningGame.getData();

                    fitness += runningGame.getFitnessScore();
                    fitnesses.append(fitness);
                    runningGame.close();
                fitness = runnerConfig.fitnessFromArray(fitnesses);
                genome.fitness = fitness;
        

    #parallel versions of eval_genomes_feedforward - DUMMY FUNCTIONS, should never be passed to a parallel process; pass the GenomeExecutor function itself
    def eval_genome_batch_feedforward(self,genomes,config,processNum):
        return GenomeExecutor.eval_genome_batch_feedforward(config,self.runConfig,self.game,genomes,None)[1];
    
    def eval_training_data_batch_feedforward(self,genomes,config,data):
        return GenomeExecutor.eval_training_data_batch_feedforward(config,self.runConfig,self.game,genomes,data,None)[1];

    #evaluate a population with the game as a feedforward neural net
    def eval_genomes_feedforward(self, genomes, config):
        for genome_id,genome in genomes:
            genome.fitness = 0; #sanity check
        if (self.runConfig.training_data is None):
            if (self.runConfig.parallel):
                self.pool.send_message('start_generation',self.runConfig,self.generation,send_on_start=True);

                chunkFactor = 4;
                if hasattr(self.runConfig,'chunkFactor') and self.runConfig.chunkFactor is not None:
                    chunkFactor = self.runConfig.chunkFactor;
                
                chunkSize,extra  = divmod(len(genomes),self.runConfig.parallel_processes * chunkFactor);
                if extra:
                    chunkSize += 1;

                print(f'Starting parallel processing for {len(genomes)} evals over {self.runConfig.parallel_processes} processes');

                fitnesses = {};
                for x in tqdm(self.pool.imap_unordered('map_eval_genome_feedforward',[(gen,gid) for gid,gen in genomes],chunksize=chunkSize),total=len(genomes)):
                    if (isinstance(x,Exception)):
                        raise x;
                    id,fitness = x;
                    fitnesses[id]=fitness;
                for genome_id,genome in genomes:
                    genome.fitness += fitnesses[genome_id];
            else:
                for genome_id, genome in tqdm(genomes):
                    genome.fitness += self.eval_genome_feedforward(genome,config)
        else:
            if (self.runConfig.parallel):
                self.pool.send_message('start_generation',self.runConfig,self.generation,genomes);
                genomes = genomes;
                tdata = self.runConfig.training_data.active_data;

                chunkFactor = 4;
                if hasattr(self.runConfig,'chunkFactor') and self.runConfig.chunkFactor is not None:
                    chunkFactor = self.runConfig.chunkFactor;
                
                chunkSize,extra = divmod(len(tdata), self.runConfig.parallel_processes * chunkFactor);
                if extra:
                    chunkSize += 1;
                chunkSize = 1;
                print(f'Starting parallel processing for {len(genomes)*len(tdata)} evals over {self.runConfig.parallel_processes} processes');

                datum_fitnesses = {};
                for x in tqdm(self.pool.imap_unordered('map_eval_genomes_feedforward',[(id,id) for id in tdata],chunksize=chunkSize),total=len(tdata)):
                    if (isinstance(x,Exception)):
                        raise x;
                    id,fitnesses = x;
                    # print('id completed:',id);
                    datum_fitnesses[id] = fitnesses;
                
                if hasattr(self.runConfig,"saveFitness") and self.runConfig.saveFitness:
                    self.fitness_reporter.save_data(datum_fitnesses);
                
                for fitnesses in datum_fitnesses.values():
                    for genome_id,genome in genomes:
                        genome.fitness += fitnesses[genome_id];
            else:
                if hasattr(self.runConfig,"saveFitness") and self.runConfig.saveFitness:
                    fitness_data = {};
                    for did in tqdm(self.runConfig.training_data.active_data):
                        fitnesses = {};
                        for genome_id, genome in tqdm(genomes):
                            fitness = self.eval_genome_feedforward(genome,config,trainingDatumId=did)
                            fitnesses[genome_id] = fitness;
                            genome.fitness += fitness;
                        fitness_data[did] = fitnesses;
                    self.fitness_reporter.save_data(fitness_data);
                else:
                    for did in tqdm(self.runConfig.training_data.active_data):
                        for genome_id, genome in tqdm(genomes):
                            genome.fitness += self.eval_genome_feedforward(genome,config,trainingDatumId=did)                



    def eval_genome_feedforward(self,genome,config,trainingDatumId:int=None):
        return GenomeExecutor.eval_genome_feedforward(genome,config,self.runConfig,self.game,trainingDatumId=trainingDatumId);
      

class GenomeExecutorException(Exception): 
    def __init__(self,e):
        self.underlying = e;

class GenomeExecutorPool(Pool):
    def __init__(self,*args,**kwargs):
        self._actor_pool:list[tuple[GenomeExecutor,int]];
        self._start_msg = None;
        self._start_args = None;
        if 'initializer' in kwargs:
            kwargs.pop('initializer');
        super().__init__(*args,**kwargs);

    def _new_actor_entry(self):
        # NOTE(edoakes): The initializer function can't currently be used to
        # modify the global namespace (e.g., import packages or set globals)
        # due to a limitation in cloudpickle.
        # Cache the PoolActor with options
        if not self._pool_actor:
            self._pool_actor = GenomeExecutor.options(**self._ray_remote_args)
        actor = self._pool_actor.remote(self._initializer, self._initargs);
        if self._start_msg is not None:
            getattr(actor,self._start_msg).remote(*self._start_args[0],**self._start_args[1]);
        return (actor, 0);

    def send_message(self,msg_func:str,*args,block=False,send_on_start=False,**kwargs): #NOTE: should only be called if actors don't have any outstanding tasks, don't know if there's a good way to check for that
        if send_on_start:
            self._start_msg = msg_func;
            self._start_args = (args,kwargs);
        refs = [getattr(actor[0],msg_func).remote(*args,**kwargs) for actor in self._actor_pool];
        return ray.get(refs) if block else refs;

        

#Genome Executor: Class that handles any and all genome processing, packaged and globalized for easier interface with parallelism
#Ok, so
#WHY does this exist, you may ask?
#This is pretty much entirely for multiprocessing reasons. These functions used to be part of the game_runner_neat class, but there ended up being a lot of pickling overhead, and - more importantly - process id assignment requires global variables. 
#Since global variables are hard and dumb, I use class variables and class methods instead. Basically the same thing, but still encapsulated.
#These functions were almost entirely cut&pasted from the above class, and the functions were aliased for backwards compatibility
@ray.remote
class GenomeExecutor:
    """Actor used to process tasks submitted to a Pool."""

    def __init__(self, initializer=None, initargs=None):
        initargs = initargs or ()
        self.initial = self.initProcess(*initargs)

    def ping(self):
        # Used to wait for this actor to be initialized.
        pass

    def run_batch(self, func:str, batch):
        print("running function batch on",func);
        results = []
        f = getattr(self,func,None);
        assert isinstance(f,Callable);
        for args, kwargs in batch:
            args = args or ()
            kwargs = kwargs or {}
            try:
                results.append(f(*args, **kwargs))
            except Exception as e:
                results.append(GenomeExecutorException(e))
        return results


    CHECKPOINT_INTERVAL = 0; # interval <= 0 means no checkins
    def initProcess(self,id_queue:Queue,game:EvalGame,neat_config):
        self.pnum = id_queue.get();
        self.game = game;
        print(f"process {self.pnum} started");
        # eGame.gameClass.initProcess(self.pnum,eGame);
        self.count = 0;
        self.generation = None;
        self.last_checkpoint_time = None;
        self.neat_config = neat_config;
        if asizeof:
            print("init process: total args size:",asizeof.asizesof(id_queue,game,neat_config));

    def start_generation(self,runConfig:RunnerConfig,generation:int,genomes:list[tuple[int,Any]]|None=None):
        if asizeof:
            print("start generation: total args size:",asizeof.asizesof(runConfig,generation,genomes));
        self.runConfig = runConfig;
        self.generation = generation;
        self.genomes = genomes;

    def eval_genome_batch_feedforward(self,config,runnerConfig,game,genomes,return_id,gen=None):
        try:
            if gen is not None:
                if gen != self.generation:
                    self.count = 0;
                self.generation = gen;
            fitnesses:dict[int,float] = {genome_id:0 for genome_id,_ in genomes};
            for genome_id, genome in genomes:
                self.count += 1;
                if self.CHECKPOINT_INTERVAL > 0 and self.count % self.CHECKPOINT_INTERVAL == 0:
                    time = datetime.now()
                    print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKPOINT_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
                    self.last_checkpoint_time = time;
                fitnesses[genome_id] += self.eval_genome_feedforward(genome,config,runnerConfig,game);
            return (return_id,fitnesses);
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    def eval_training_data_batch_feedforward(self,config,runnerConfig,game,genomes,data:list[int],return_id,gen=None):
        try:
            if gen is not None:
                if gen != self.generation:
                    self.count = 0;
                self.generation = gen;
            fitnesses:dict[int,float] = {genome_id:0 for genome_id,_ in genomes};
            for datum_id in data:
                for genome_id,genome in genomes:
                    fitnesses[genome_id] += self.eval_genome_feedforward(genome,config,runnerConfig,game,trainingDatumId=datum_id);
                    self.count += 1;
                    if self.CHECKPOINT_INTERVAL > 0 and self.count % self.CHECKPOINT_INTERVAL == 0:
                        time = datetime.now()
                        print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKPOINT_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
                        self.last_checkpoint_time = time;
            return (return_id,fitnesses);
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    #map methods - iterate externally; return_id is used to recombine with i- or async- pool methods where order is not guaranteed
    #training data - eval multiple genomes; map inputs are data_id, return id (likely the same)
    def map_eval_genomes_feedforward(self,b_in:tuple[int,int]):
        datum_id,return_id = b_in
        try:
            # tracemalloc.start();
            # start = tracemalloc.take_snapshot();
            assert self.genomes is not None
            fitnesses:dict[int,float] = {genome_id:0 for genome_id,_ in self.genomes};
            
            for genome_id,genome in self.genomes:
                self.count += 1;
                if self.CHECKPOINT_INTERVAL > 0 and self.count % self.CHECKPOINT_INTERVAL == 0:
                    time = datetime.now()
                    print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKPOINT_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
                    self.last_checkpoint_time = time;
                
                fitnesses[genome_id] += self.eval_genome_feedforward(genome,self.neat_config,self.runConfig,self.game,trainingDatumId=datum_id);
                # loop = tracemalloc.take_snapshot();
                # diff = start.compare_to(loop,'traceback');
                # display_diff(diff);
                # start = loop;

            return (return_id,fitnesses);
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    #no training data; map inputs are genome, reference_id
    def map_eval_genome_feedforward(self,b_in:tuple[Any,int]):
        genome,return_id = b_in;
        try:
            self.count += 1;
            if self.CHECKPOINT_INTERVAL > 0 and self.count % self.CHECKPOINT_INTERVAL == 0:
                time = datetime.now()
                print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKPOINT_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
                self.last_checkpoint_time = time;
            return (return_id,self.eval_genome_feedforward(genome,self.neat_config,self.runConfig,self.game));
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    def eval_genome_feedforward(self,genome,config,runnerConfig:RunnerConfig,game:EvalGame,trainingDatumId=None):
        try:
            net = neat.nn.FeedForwardNetwork.create(genome,config);
            
            fitnesses:list[float] = [];
            for _ in range(runnerConfig.numTrials):
                fitness = 0;
                runningGame = None;
                if self.pnum is not None:
                    runningGame = game.start(runnerConfig,training_datum_id = trainingDatumId, process_num = self.pnum);
                else:
                    runningGame = game.start(runnerConfig,training_datum_id = trainingDatumId)
                if runnerConfig.fitness_collection_type != None and 'delta' in runnerConfig.fitness_collection_type:
                    fitness -= runningGame.getFitnessScore();

                #get the current data from the running game, as specified by the runnerConfig
                gameData = runningGame.getData();
                while (runningGame.isRunning(useCache=True)):

                    try:
                        gameInput = net.activate(gameData);
                    except:
                        print('Error in activating net with data ', gameData, ' and mapped data ', runningGame.getMappedData());
                        print('Error body: ', sys.exc_info());
                        raise Exception();

                    runningGame.tickInput(gameInput);

                    if (runnerConfig.fitness_collection_type != None and 'continuous' in runnerConfig.fitness_collection_type):
                        fitness += runningGame.getFitnessScore();
                    
                    gameData = runningGame.getData();

                # print(f"game size: {total_size(runningGame)}");

                fitness += runningGame.getFitnessScore();
                fitnesses.append(fitness);
                runningGame.close();

            fitness = runnerConfig.fitnessFromArray()(fitnesses);
            return fitness; 
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

def display_diff(diff:list[tracemalloc.StatisticDiff],limit=10):
    print("[ Top 10 differences ]")
    for stat in diff[:10]:
        print(stat)


def display_top(snapshot:tracemalloc.Snapshot, key_type='lineno', limit=10):
    try:
        snapshot = snapshot.filter_traces((
            tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
            tracemalloc.Filter(False, "<unknown>"),
        ))

        top_stats = snapshot.statistics('traceback')

        if len(top_stats) > 0:
            # pick the biggest memory block
            stat = top_stats[0]
            print("largest memory block traceback: %s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
            for line in stat.traceback.format():
                print(line)
        else:
            print("no traceback to print");

        top_stats = snapshot.statistics(key_type)

        print("Top %s lines" % limit)
        for index, stat in enumerate(top_stats[:limit], 1):
            frame = stat.traceback[0]
            print("#%s: %s:%s: %.1f KiB"
                % (index, frame.filename, frame.lineno, stat.size / 1024))
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print('    %s' % line)

        other = top_stats[limit:]
        if other:
            size = sum(stat.size for stat in other)
            print("%s other: %.1f KiB" % (len(other), size / 1024))
        total = sum(stat.size for stat in top_stats)
        print("Total allocated size: %.1f KiB" % (total / 1024))
    except e:
        print(e);
        raise e.with_traceback();
