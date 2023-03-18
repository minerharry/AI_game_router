import copy
import re
from traceback import format_tb
from typing import Any, Callable, Generic, Iterable, Type
import warnings
import ray
from ray.util.queue import _QueueActor
from tqdm import tqdm
from baseGame import EvalGame, RunGame
import neat
import tracemalloc
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from play_level import TaskFitnessReporter
from runnerConfiguration import RunnerConfig
import os.path
import os
import visualize
import sys
import random
from fitnessReporter import FitnessReporter
from datetime import datetime
import linecache
from logReporting import LoggingReporter
from renderer import Renderer as RendererReporter
try:
    from videofig import videofig as vidfig
except:
    vidfig = None;
from neat.six_util import iteritems, itervalues

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
        checkpoint_folder = 'checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+run_name.replace(' ','_');
        if manual_generation is None:
            files = os.listdir(checkpoint_folder);
            maxGen = -1;
            for file in files:
                m = re.match("run-checkpoint-([0-9]+)",file);
                if m:
                    gen = int(m.group(1));
                    if (gen>maxGen):
                        maxGen = gen;
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '/run-checkpoint-' + str(maxGen) + ".gz",config_transfer=manual_config_override);
        else:
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '/run-checkpoint-' + str(manual_generation) + ".gz",config_transfer=manual_config_override);

        return self.run(pop.config,run_name,render=render,pop=pop,single_gen=single_gen);

    def replay_generation(self,generation,run_name,render=False,genome_config_edits=None):
        checkpoint_folder = 'checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+run_name.replace(' ','_');
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '/run-checkpoint-' + str(generation) + '.gz');

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
            os.makedirs("checkpoints/games/"+self.runConfig.gameName.replace(' ','_')+f'/{self.run_name}',exist_ok=True);            
            pop.add_reporter(neat.Checkpointer(1,filename_prefix='checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+self.run_name+'/run-checkpoint-'));

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
            pool_kwargs = getattr(self.runConfig,'pool_kwargs',{});
            self.pool = GenomeExecutorPool(self.runConfig.parallel_processes, initargs=(self.game,config,idQueue),**pool_kwargs);
            

        if not single_gen or force_fitness:
            self.fitness_reporter = FitnessReporter(self.runConfig.gameName,self.run_name);
            pop.add_reporter(self.fitness_reporter);

        self.generation = pop.generation;
        
        winner = pop.run(self.eval_genomes,self.runConfig.generations if not single_gen else 1);

        return winner;

    def check_output_connections(self,generation,run_name,target_output,render=False):
        file = 'checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+run_name.replace(' ','_')+'/run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        connected = [];
        for g in pop.population.values():
            for connection in g.connections:
                if (connection[1] == target_output):
                    connected.append(g);
                    break;
        [print (connectedGenome.key) for connectedGenome in connected];

    def render_worst_genome(self,generation,run_name,net=False,override_config=None):
        file = 'checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+run_name.replace(' ','_')+'/run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        config = override_config or pop.config;
        worst = None
        worst_id = -1;
        for id,g in pop.population.items():
            if worst is None or g.fitness < worst.fitness:
                worst = g
                worst_id = id;
        if worst is not None:
            self.render_genome(worst_id,worst,config,net=net);
        else:
            raise Exception("no genomes to eval");

    def render_genome_by_id(self,genomeId,generation,run_name,net=False,override_config=None):
        file = 'checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+run_name.replace(' ','_')+'/run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        genome = None;
        for g in itervalues(pop.population):
            if g.key == genomeId:
                genome = g;
                break;
        config = override_config or pop.config;
        self.render_genome(genomeId,genome,config,net=net);
                    
    def render_custom_genome_object(self,obj,config,net=False):
        self.render_genome(None,obj,config,net=net)

    def replay_best(self,generation,run_name,net=False,randomReRoll=False,number=1,override_config=None):
        if number < 1:
            raise Exception("must replay at least one genome");
        file = 'checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+run_name.replace(' ','_')+'/run-checkpoint-' + str(generation) + ".gz";
        pop = neat.Checkpointer.restore_checkpoint(file);
        config = override_config or pop.config;
        #self.eval_genomes(list(iteritems(pop.population)),config);
        if (randomReRoll):
            random.seed();
        sort = sorted(pop.population.items(),key=lambda x: x[0]);
        for gid,g in sort[:number]:
            self.render_genome(gid,g,config,net=net);

    def run_top_genomes(self,generation,run_name,number,doFitness=False,randomReRoll=False,override_config=None):
        checkpoint_folder = 'checkpoints/games/'+self.runConfig.gameName.replace(' ','_')+'/'+run_name.replace(' ','_');
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '/run-checkpoint-' + str(generation) + '.gz');

        config = override_config or pop.config;

        if self.runConfig.parallel:
            idQueue = Queue();
            [idQueue.put(i) for i in range(self.runConfig.parallel_processes)];
            pool_kwargs = getattr(self.runConfig,'pool_kwargs',{});
            self.pool = GenomeExecutorPool(self.runConfig.parallel_processes, initargs=(self.game,config,idQueue),**pool_kwargs);
            
        self.run_name = run_name.replace(' ','_');
        if doFitness:
            self.fitness_reporter = FitnessReporter(self.runConfig.gameName,self.run_name + f"_top_{number}");
            self.fitness_reporter.start_generation(generation);

        if (randomReRoll):
            random.seed();

        sort = sorted(pop.population.items(),key=lambda x: x[0]);

        self.eval_genomes(sort[:number],config);
        

    def render_genome(self,genome_id:int|None,genome:Any,config:Any,net=False):
        if (net):
            flattened_data = self.runConfig.flattened_return_data();
            shaped_data = self.runConfig.return_data_shape();
            visualize.draw_net(config,genome,view=True,node_names=dict([(-i-1,flattened_data[i]) for i in range(len(flattened_data))]),nodes_shape=shaped_data);
        if self.runConfig.training_data is None:
            if (self.runConfig.recurrent):  
                self.render_genome_recurrent(genome_id,genome,config,net=net);
            else:
                self.render_genome_feedforward(genome_id,genome,config,net=net);
        else:
            with self.runConfig.training_data.poll_data(self.generation) as tdata:
                for did in tdata:
                    if (self.runConfig.recurrent):  
                        self.render_genome_recurrent(genome_id,genome,config,net=False,training_datum_id = did);
                    else:
                        self.render_genome_feedforward(genome_id,genome,config,net=False,training_datum_id = did);



    #render a genome with the game as a recurrent neural net
    def render_genome_recurrent(self, genome_id:int|None,genome:Any,config:Any,*args,net=False,**kwargs):
        if vidfig is None:
            warnings.warn("Unable to render genome: visualization not available due to a missing optional dependency (matplotlib)")
            return
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
            runningGame = self.game.start(runnerConfig,genome_id=genome_id);
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
    def render_genome_feedforward(self, genome_id:int|None,genome:Any, config,net=False,training_datum_id=None):
        GenomeExecutor.Executor(self.game,config,self.runConfig,{genome_id or 0:genome},generation=self.generation).render_genome_feedforward(genome_id or 0,trainingDatumId=training_datum_id,render_net=net)
        # runnerConfig = self.runConfig;
        # if (net):
        #     flattened_data = runnerConfig.flattened_return_data();
        #     shaped_data = runnerConfig.return_data_shape();
        #     visualize.draw_net(config,genome,view=True,node_names=dict([(-i-1,flattened_data[i]) for i in range(len(flattened_data))]),nodes_shape=shaped_data);
        
        # if (runnerConfig.parallel and False):
        #     return;
        #     #TODO: implement parallel game processing
        # else:
            
        #     net = neat.nn.FeedForwardNetwork.create(genome,config);
        #     runningGame = self.game.start(runnerConfig,genome_id=genome_id,training_datum = training_datum);
        #     images = [];
        #     fitness = 0;
        #     if 'delta' in runnerConfig.fitness_collection_type:
        #         fitness -= runningGame.getFitnessScore();

        #     max_fitness = 0;
        #     if 'max' in runnerConfig.fitness_collection_type:
        #         max_fitness = runningGame.getFitnessScore();
        #     #get the current inputs from the running game, as specified by the runnerConfig
        #     gameData = runningGame.getData();
        #     while (runningGame.isRunning(useCache=True)):
        
        #         gameInput = net.activate(gameData);

        #         if (self.runConfig.external_render):
        #             images.append(runningGame.tickRenderInput(gameInput));
        #         else:
        #             runningGame.tickRenderInput(gameInput);


        #         if ('continuous' in runnerConfig.fitness_collection_type):
        #             fitness += runningGame.getFitnessScore();
        #         elif ('max' in runnerConfig.fitness_collection_type):
        #             max_fitness = max(max_fitness,runningGame.getFitnessScore());

        #         gameData = runningGame.getData();

        #     if 'max' in runnerConfig.fitness_collection_type:
        #         fitness += max_fitness;
        #     elif 'continuous' not in runnerConfig.fitness_collection_type: #prevent double counting
        #         fitness += runningGame.getFitnessScore();

        #     print('final genome fitness: ' + str(fitness));

                        
        #     runningGame.close();
        #     if (self.runConfig.external_render):
        #         get_genome_frame.images = images;
        #         get_genome_frame.initialized = False;
        #         vidfig(len(images),get_genome_frame,play_fps=runnerConfig.playback_fps);


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
    def eval_genome_batch_feedforward(self,genomes:list[tuple[int,Any]],config,processNum):
        return GenomeExecutor.Executor(self.game,config,self.runConfig,genomes).eval_genome_batch_feedforward([l[0] for l in genomes])[1];
    
    def eval_training_data_batch_feedforward(self,genomes,config,data):
        return GenomeExecutor.Executor(self.game,config,self.runConfig,genomes).eval_training_data_batch_feedforward([l[0] for l in genomes],data)[1];

    #evaluate a population with the game as a feedforward neural net
    def eval_genomes_feedforward(self, genomes, config):
        for genome_id,genome in genomes:
            genome.fitness = 0; #sanity check
        if (self.runConfig.parallel):
            if (self.runConfig.training_data is None):
                self.pool.send_message('start_generation',self.runConfig,self.generation,genomes,send_on_start=True);

                chunkFactor = 4;
                if hasattr(self.runConfig,'chunkFactor') and self.runConfig.chunkFactor is not None:
                    chunkFactor = self.runConfig.chunkFactor;
                
                chunkSize,extra  = divmod(len(genomes),self.runConfig.parallel_processes * chunkFactor);
                if extra:
                    chunkSize += 1;

                print(f'Starting parallel processing for {len(genomes)} evals over {self.runConfig.parallel_processes} processes');

                fitnesses = {};
                for x in tqdm(self.pool.imap_unordered('map_eval_genome_feedforward',[gid for gid in genomes],chunksize=chunkSize),total=len(genomes)):
                    if (isinstance(x,Exception)):
                        raise x;
                    id,fitness = x;
                    fitnesses[id]=fitness;
                for genome_id,genome in genomes:
                    genome.fitness += fitnesses[genome_id];
            else:
                with self.runConfig.training_data.poll_data(self.generation) as tdata:
                    self.pool.send_message('start_generation',self.runConfig,self.generation,genomes);
                    genomes = genomes;

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
            executor = GenomeExecutor.Executor(self.game,config,self.runConfig,genomes,generation=self.generation)
            if (self.runConfig.training_data is None):
                for genome_id, genome in tqdm(genomes):
                    genome.fitness += executor.eval_genome_feedforward(genome_id);                
            else:
                with self.runConfig.training_data.poll_data(self.generation) as tdata:
                    if hasattr(self.runConfig,"saveFitness") and self.runConfig.saveFitness:
                        fitness_data = {};
                        for did in tqdm(tdata):
                            fitnesses = {};
                            for genome_id, genome in tqdm(genomes):
                                fitness = executor.eval_genome_feedforward(genome_id,did);
                                fitnesses[genome_id] = fitness;
                                genome.fitness += fitness;
                            fitness_data[did] = fitnesses;
                        self.fitness_reporter.save_data(fitness_data);
                    else:
                        for did in tqdm(tdata):
                            for genome_id, genome in tqdm(genomes):
                                genome.fitness += executor.eval_genome_feedforward(genome_id,did);                
      

class GenomeExecutorException(BaseException): 
    def __init__(self,e:BaseException,traceback=None):
        self.__context__ = e;
        self.tb = traceback;
    
    def __str__(self):
        return f"underlying exception {self.__context__} thrown during genomeexecutor execution" + (f" with traceback {self.tb}" if self.tb else None)

class GenomeExecutorPool(Pool[str]):
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
            self._pool_actor = GenomeExecutorActor.options(**self._ray_remote_args)
        actor = self._pool_actor.remote(*self._initargs);
        if self._start_msg is not None:
            assert self._start_args is not None
            getattr(actor,self._start_msg).remote(*self._start_args[0],**self._start_args[1]);
        return (actor, 0);

    def send_message(self,msg_func:str,*args,block=False,send_on_start=False,**kwargs): #NOTE: should only be called if actors don't have any outstanding tasks, don't know if there's a good way to check for that
        if send_on_start:
            self._start_msg = msg_func;
            self._start_args = (args,kwargs);
            assert self._start_args is not None
        refs = [getattr(actor[0],msg_func).remote(*args,**kwargs) for actor in self._actor_pool];
        return ray.get(refs) if block else refs;



#Genome Executor: Class that handles any and all genome processing, packaged and globalized for easier interface with parallelism
#Ok, so
#WHY does this exist, you may ask?
#This is pretty much entirely for multiprocessing reasons. These functions used to be part of the game_runner_neat class, but there ended up being a lot of pickling overhead, and - more importantly - process id assignment requires global variables. 
#Since global variables are hard and dumb, I use class variables and class methods instead. Basically the same thing, but still encapsulated.
#These functions were almost entirely cut&pasted from the above class, and the functions were aliased for backwards compatibility
class GenomeExecutor:
    """Actor used to process tasks submitted to a Pool."""

    @classmethod
    def Executor(cls,game:EvalGame,neat_config:Any,runConfig:RunnerConfig,genomes:dict[int,Any]|list[tuple[int,Any]],process_num:int|None=None,generation:int|None=None):
        ex = GenomeExecutor(game,neat_config,process_num=process_num);
        ex.start_generation(runConfig,generation,genomes);
        return ex;

    def __init__(self,game:EvalGame,neat_config,process_num=None):
        self.initProcess(game,neat_config,process_num=process_num)

    CHECKIN_INTERVAL = 0; # interval <= 0 means no checkins
    def initProcess(self,game:EvalGame,neat_config,process_num=None):
        self.pnum = process_num;
        self.game = game;
        if self.pnum is not None:
            print(f"process {self.pnum} started");
        self.count = 0;
        self.generation = None;
        self.last_checkpoint_time = None;
        self.neat_config = neat_config;
        # self.game.gameClass.initProcess(self.pnum,self.game);

    def start_generation(self,runConfig:RunnerConfig,generation:int|None,genomes:dict[int,Any]|list[tuple[int,Any]]):
        self.runConfig = runConfig;
        self.generation = generation;
        if isinstance(genomes,list):
            genomes = dict(genomes);
        self.genomes = genomes;
    
    ### ray methods
    def ping(self):
        # Used to wait for this actor to be initialized.
        pass

    def run_batch(self, func:str, batch):
        # print("running function batch on",func);
        results = []
        f = getattr(self,func,None);
        assert isinstance(f,Callable);
        for args, kwargs in batch:
            args = args or ()
            kwargs = kwargs or {}
            try:
                results.append(f(*args, **kwargs))
            except Exception as e:
                results.append(GenomeExecutorException(e,format_tb(e.__traceback__)))
        return results


    ### execution methods 

    def eval_genome_batch_feedforward(self,genome_ids:Iterable[int],return_id:Any=None,gen=None):
        try:
            if gen is not None:
                #cringe this should not be used most likely
                if gen != self.generation:
                    self.count = 0;
                self.generation = gen;
            fitnesses:dict[int,float] = {genome_id:0 for genome_id in genome_ids};
            for genome_id in genome_ids:
                self.count += 1;
                if self.CHECKIN_INTERVAL > 0 and self.count % self.CHECKIN_INTERVAL == 0:
                    time = datetime.now()
                    print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKIN_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
                    self.last_checkpoint_time = time;
                fitnesses[genome_id] += self.eval_genome_feedforward(genome_id);
            return (return_id,fitnesses);
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    def eval_training_data_batch_feedforward(self,genome_ids:Iterable[int],data:Iterable[int],return_id:Any=None,gen=None):
        try:
            if gen is not None:
                if gen != self.generation:
                    self.count = 0;
                self.generation = gen;
            fitnesses:dict[int,float] = {genome_id:0 for genome_id in genome_ids};
            for datum_id in data:
                for genome_id in genome_ids:
                    fitnesses[genome_id] += self.eval_genome_feedforward(genome_id,trainingDatumId=datum_id);
                    self.count += 1;
                    if self.CHECKIN_INTERVAL > 0 and self.count % self.CHECKIN_INTERVAL == 0:
                        time = datetime.now()
                        print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKIN_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
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
            fitnesses:dict[int,float] = {genome_id:0 for genome_id in self.genomes};
            
            for genome_id in self.genomes:
                self.count += 1;
                if self.CHECKIN_INTERVAL > 0 and self.count % self.CHECKIN_INTERVAL == 0:
                    time = datetime.now()
                    print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKIN_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
                    self.last_checkpoint_time = time;
                
                fitnesses[genome_id] += self.eval_genome_feedforward(genome_id,trainingDatumId=datum_id);
                # loop = tracemalloc.take_snapshot();
                # diff = start.compare_to(loop,'traceback');
                # display_diff(diff);
                # start = loop;

            return (return_id,fitnesses);
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    #no training data; map inputs are genome, reference_id
    def map_eval_genome_feedforward(self,b_in:tuple[int,Any,int]):
        genome_id,genome,return_id = b_in;
        try:
            self.count += 1;
            if self.CHECKIN_INTERVAL > 0 and self.count % self.CHECKIN_INTERVAL == 0:
                time = datetime.now()
                print(f'Parallel Checkpoint - Process #{self.pnum} at {time}' + ('' if self.generation is None else f'; Count: {self.count} evals completed this generation ({self.generation})') + ('' if self.last_checkpoint_time is None else f'; Eval Speed: {self.CHECKIN_INTERVAL/(time-self.last_checkpoint_time).total_seconds():.5f}'));
                self.last_checkpoint_time = time;
            return (return_id,self.eval_genome_feedforward(genome_id));
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    def eval_genome_feedforward(self,genome_id:int,trainingDatumId=None):
        try:
            genome = self.genomes[genome_id];
            net = neat.nn.FeedForwardNetwork.create(genome,self.neat_config);
            
            fitnesses:list[float] = [];
            for _ in range(self.runConfig.numTrials):
                fitness = 0;
                runningGame = None;
                if self.pnum is not None:
                    runningGame = self.game.start(self.runConfig,genome_id=genome_id,training_datum_id = trainingDatumId, process_num = self.pnum);
                else:
                    runningGame = self.game.start(self.runConfig,genome_id=genome_id,training_datum_id = trainingDatumId)
                
                max_fitness = None
                if self.runConfig.fitness_collection_type != None:
                    if 'delta' in self.runConfig.fitness_collection_type:
                        fitness -= runningGame.getFitnessScore();
                    if 'max' in self.runConfig.fitness_collection_type:
                        max_fitness = runningGame.getFitnessScore();

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

                    if self.runConfig.fitness_collection_type != None:
                        if 'continuous' in self.runConfig.fitness_collection_type:
                            fitness += runningGame.getFitnessScore();
                        if max_fitness is not None:
                            max_fitness = max(max_fitness,runningGame.getFitnessScore());
                                        
                    gameData = runningGame.getData();

                if self.runConfig.fitness_collection_type != None:
                    if max_fitness is not None:
                        fitness += max_fitness;
                    if 'continuous' in self.runConfig.fitness_collection_type:
                        fitness -= runningGame.getFitnessScore();
                
                fitness += runningGame.getFitnessScore();
                fitnesses.append(fitness);
                runningGame.close();

            fitness = self.runConfig.fitnessFromArray(fitnesses);
            return fitness; 
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

    def render_genome_feedforward(self,genome_id:int,trainingDatumId=None,render_net:bool=False):
        try:
            genome = self.genomes[genome_id];
            net = neat.nn.FeedForwardNetwork.create(genome,self.neat_config);

            if render_net:
                flattened_data = self.runConfig.flattened_return_data();
                shaped_data = self.runConfig.return_data_shape();
                visualize.draw_net(self.neat_config,genome,view=True,node_names=dict([(-i-1,flattened_data[i]) for i in range(len(flattened_data))]),nodes_shape=shaped_data)
            
            fitnesses:list[float] = [];
            for _ in range(self.runConfig.numTrials):
                frames = [];
                fitness = 0;
                runningGame = None;
                if self.pnum is not None:
                    runningGame = self.game.start(self.runConfig,genome_id=genome_id,training_datum_id = trainingDatumId, process_num = self.pnum);
                else:
                    runningGame = self.game.start(self.runConfig,genome_id=genome_id,training_datum_id = trainingDatumId)
                
                max_fitness = None
                if self.runConfig.fitness_collection_type != None:
                    if 'delta' in self.runConfig.fitness_collection_type:
                        fitness -= runningGame.getFitnessScore();
                    if 'max' in self.runConfig.fitness_collection_type:
                        max_fitness = runningGame.getFitnessScore();

                #get the current data from the running game, as specified by the runnerConfig
                gameData = runningGame.getData();
                while (runningGame.isRunning(useCache=True)):
                    try:
                        gameInput = net.activate(gameData);
                    except:
                        print('Error in activating net with data ', gameData, ' and mapped data ', runningGame.getMappedData());
                        print('Error body: ', sys.exc_info());
                        raise Exception();
                
                    if self.runConfig.external_render:
                        frames.append(runningGame.tickRenderInput(gameInput));
                    else:
                        runningGame.tickRenderInput(gameInput)

                    if self.runConfig.fitness_collection_type != None:
                        if 'continuous' in self.runConfig.fitness_collection_type:
                            fitness += runningGame.getFitnessScore();
                        if max_fitness is not None:
                            max_fitness = max(max_fitness,runningGame.getFitnessScore());
                                        
                    gameData = runningGame.getData();

                if self.runConfig.fitness_collection_type != None:
                    if max_fitness is not None:
                        fitness += max_fitness;
                    if 'continuous' in self.runConfig.fitness_collection_type:
                        fitness -= runningGame.getFitnessScore();
                
                fitness += runningGame.getFitnessScore();
                fitnesses.append(fitness);
                runningGame.close();
            
                if (self.runConfig.external_render):
                    get_genome_frame.images = frames;
                    get_genome_frame.initialized = False;
                    vidfig(len(images),get_genome_frame,play_fps=runnerConfig.playback_fps);



            fitness = self.runConfig.fitnessFromArray(fitnesses);
            return fitness; 
        except KeyboardInterrupt:
            raise GenomeExecutorException(KeyboardInterrupt());

@ray.remote(num_cpus=1)
class GenomeExecutorActor(GenomeExecutor):
    def __init__(self, game: EvalGame, neat_config, id_queue:Queue):
        super().__init__(game, neat_config, process_num = id_queue.get())





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
    except Exception as e:
        print(e);
        raise e;
