import neat
import baseGame
import runnerConfiguration
import os.path
import os
import visualize
import sys
import random
import numpy as np
import functools
from datetime import datetime
#import concurrent.futures
import multiprocessing
from logReporting import LoggingReporter
from renderer import Renderer as RendererReporter
from videofig import videofig as vidfig
from neat.six_util import iteritems, itervalues
try:
    from pympler import tracker
except:
    pass;


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
    def __init__(self,game,runnerConfig):
        self.game = game;
        self.runConfig = runnerConfig;
        self.generation = None;

    def continue_run(self,run_name,render=False,manual_generation=None):
        checkpoint_folder = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_');
        if manual_generation is None:
            files = os.listdir(checkpoint_folder);
            maxGen = -1;
            for file in files:
                if (int(file.split('run-checkpoint-')[1])>maxGen):
                    maxGen = int(file.split('run-checkpoint-')[1]);
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '\\run-checkpoint-' + str(maxGen));
        else:
            pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '\\run-checkpoint-' + str(manual_generation));

        return self.run(pop.config,run_name,render=render,pop=pop);

    def replay_generation(self,generation,run_name,render=False,genome_config_edits=None):
        checkpoint_folder = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_');
        pop = neat.Checkpointer.restore_checkpoint(checkpoint_folder + '\\run-checkpoint-' + str(generation));

        config = pop.config;

        if (genome_config_edits is not None):
            for k,v in genome_config_edits:
                if hasattr(config.genome_config,k):
                    setattr(config.genome_config,k,v);

        return self.run(config,run_name,render=render,pop=pop,single_gen=True);

    def run(self,config,run_name,render=False,pop=None,single_gen=False):
        self.run_name = run_name.replace(' ','_');
        if (pop is None):
            pop = neat.Population(config);
            continuing = False;
            print(pop.population[1]);
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
            
            # if (not(os.path.exists('checkpoints'))):
            #     os.mkdir('checkpoints');
            # if (not(os.path.exists('checkpoints\\games'))):
            #     os.mkdir('checkpoints\\games');
            # if (not(os.path.exists('checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')))):
            #     os.mkdir('checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_'));
            # if (not(os.path.exists('checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')))):
            #     os.mkdir();
            
            pop.add_reporter(neat.Checkpointer(1,filename_prefix='checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+self.run_name+'\\run-checkpoint-'));

        if (render):
            pop.add_reporter(RendererReporter(self));
        if (continuing):
            pop.complete_generation();
        
        if self.runConfig.parallel:
            manager = multiprocessing.Manager()
            idQueue = manager.Queue()
            [idQueue.put(i) for i in range(self.runConfig.parallel_processes)];
            self.pool = multiprocessing.pool.Pool(self.runConfig.parallel_processes, Genome_Executor.initProcess,(idQueue,self.game.gameClass));

        self.generation = pop.generation;
        
        winner = pop.run(self.eval_genomes,self.runConfig.generations if not single_gen else 1);

        return winner;

    def check_output_connections(self,generation,config,run_name,target_output,render=False):
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation);
        pop = neat.Checkpointer.restore_checkpoint(file);
        connected = [];
        for g in itervalues(pop.population):
            for connection in g.connections:
                if (connection[1] == target_output):
                    connected.append(g);
                    break;
        [print (connectedGenome) for connectedGenome in connected];

    def render_worst_genome(self,generation,config,run_name,net=False):
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation);
        pop = neat.Checkpointer.restore_checkpoint(file);
        worst = None
        for g in itervalues(pop.population):
            if worst is None or g.fitness < worst.fitness:
                worst = g
        self.render_genome_by_id(worst.key,generation,config,run_name,net=net);

    def render_genome_by_id(self,genomeId,generation,config,run_name,net=False):
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation);
        pop = neat.Checkpointer.restore_checkpoint(file);
        genome = None;
        for g in itervalues(pop.population):
            if g.key == genomeId:
                genome = g;
                break;
        self.render_genome(genome,config,net=net);
                    
                

    def replay_best(self,generation,config,run_name,net=False,randomReRoll=False):
        file = 'checkpoints\\games\\'+self.runConfig.gameName.replace(' ','_')+'\\'+run_name.replace(' ','_')+'\\run-checkpoint-' + str(generation);
        pop = neat.Checkpointer.restore_checkpoint(file);
        #self.eval_genomes(list(iteritems(pop.population)),config);
        if (randomReRoll):
            random.seed();
        best = None
        for g in itervalues(pop.population):
            if best is None or g.fitness > best.fitness:
                best = g
        print(best);
        self.render_genome(best,config,net=net);
        

    def render_genome(self,genome,config,net=False):
        if self.runConfig.training_data is None:
            if (self.runConfig.recurrent):  
                self.render_genome_recurrent(genome,config,net=net);
            else:
                self.render_genome_feedforward(genome,config,net=net);
        else:
            if (net):
                flattened_data = self.runConfig.flattened_return_data();
                shaped_data = self.runConfig.return_data_shape();
                visualize.draw_net(config,genome,view=True,node_names=dict([(-i-1,flattened_data[i]) for i in range(len(flattened_data))]),nodes_shape=shaped_data);
            
            for datum in self.runConfig.training_data:
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
            while (runningGame.isRunning()):
                    #get the current inputs from the running game, as specified by the runnerConfig
                gameData = runningGame.getData();

                gameInput = net.advance(gameData, time_const, time_const);

                images.append(runningGame.renderInput(gameInput));

                        
            runningGame.close();
            get_genome_frame.images = images;
            get_genome_frame.initialized = False;
            vidfig(len(images),get_genome_frame,play_fps=runnerConfig.playback_fps);

        

    #render a genome with the game as a feedforward neural net
    def render_genome_feedforward(self, genome, config,net=False,training_datum=None):
        fitness_list = [];
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
            while (runningGame.isRunning()):
                #get the current inputs from the running game, as specified by the runnerConfig
                gameData = runningGame.getData();

                gameInput = net.activate(gameData);

                fitness_list.append(runningGame.getFitnessScore());

                if (self.runConfig.external_render):
                    images.append(runningGame.renderInput(gameInput));
                else:
                    runningGame.renderInput(gameInput);


                        
            runningGame.close();
            if (self.runConfig.external_render):
                get_genome_frame.images = images;
                get_genome_frame.initialized = False;
                vidfig(len(images),get_genome_frame,play_fps=runnerConfig.playback_fps);
            self.runConfig.fitness_list.append(fitness_list);

            
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
                    while (runningGame.isRunning()):
                        #get the current inputs from the running game, as specified by the runnerConfig
                        gameData = runningGame.getData();

                        gameInput = net.advance(gameData, time_const, time_const);
                        
                        runningGame.processInput(gameInput);
                        if (runnerConfig.fitness_collection_type != None and runnerConfig.fitness_collection_type == 'continuous'):
                            fitness += runningGame.getFitnessScore();
                    fitness += runningGame.getFitnessScore();
                    fitnesses.append(fitness);
                    runningGame.close();
                fitness = runnerConfig.fitnessFromArray(fitnesses);
                genome.fitness = fitness;
        

    #parallel versions of eval_genomes_feedforward - DUMMY FUNCTIONS, should never be passed to a parallel process; pass the Genome_Executor function itself
    def eval_genome_batch_feedforward(self,genomes,config,processNum):
        return Genome_Executor.eval_genome_batch_feedforward(config,self.runConfig,self.game,genomes);
    
    def eval_training_data_batch_feedforward(self,genomes,config,data):
        return Genome_Executor.eval_training_data_batch_feedforward(config,self.runConfig,self.game,genomes,data);

    #evaluate a population with the game as a feedforward neural net
    def eval_genomes_feedforward(self, genomes, config):
        for genome_id,genome in genomes:
            genome.fitness = 0; #sanity check
        if (self.runConfig.training_data is None):
            if (self.runConfig.parallel):

                batch_func = functools.partial(Genome_Executor.map_eval_genome_feedforward,config,self.runConfig,self.game,gen=self.generation);
                
                chunkFactor = 4;
                if hasattr(self.runConfig,'chunkFactor') and self.runConfig.chunkFactor is not None:
                    chunkFactor = self.runConfig.chunkFactor;
                
                chunkSize,extra  = divmod(len(genomes),self.runConfig.parallel_processes * chunkFactor);

                if extra:
                    chunkSize += 1;

                print(f'Starting parallel processing for {len(genomes)} evals over {self.runConfig.parallel_processes} processes');

                fitnesses = self.pool.map(batch_func,genomes,chunksize=chunkSize);
                for genome_id,genome in genomes:
                    genome.fitness += fitnesses[genome_id];
                # processes = [];
                # genome_batches = np.array_split(genomes,self.runConfig.parallel_processes);
                # for i in range(runConfig.parallel_processes):
                #     process = multiprocessing.Process(target=self.eval_genome_batch_feedforward,args=(genome_batches[i],config,i));
                #     processes.append(process);
                # for process in processes:
                #     process.start();
                # for process in processes:
                #     process.join();
                # return;
            else:
                for genome_id, genome in genomes:
                    genome.fitness += self.eval_genome_feedforward(genome,config)
        else:
            if (self.runConfig.parallel):
                
                #data_batches = np.array_split(self.runConfig.training_data,self.runConfig.parallel_processes);

                batch_func = functools.partial(Genome_Executor.map_eval_genomes_feedforward,config,self.runConfig,self.game,genomes,gen=self.generation);

                chunkFactor = 4;
                if hasattr(self.runConfig,'chunkFactor') and self.runConfig.chunkFactor is not None:
                    chunkFactor = self.runConfig.chunkFactor;
                
                chunkSize,extra = divmod(len(self.runConfig.training_data), self.runConfig.parallel_processes * chunkFactor);
                if extra:
                    chunkSize += 1;
                print(f'Starting parallel processing for {len(genomes)*len(self.runConfig.training_data)} evals over {self.runConfig.parallel_processes} processes');

                datum_fitnesses = self.pool.map(batch_func,self.runConfig.training_data,chunksize=chunkSize);
                for fitnesses in datum_fitnesses:
                    for genome_id,genome in genomes:
                        genome.fitness += fitnesses[genome_id];

                if hasattr(self.runConfig,"saveFitness") and self.runConfig.saveFitness:
                    os.makedirs(f"memories\\{self.runConfig.gameName.replace(' ','_')}\\{self.run_name}_history",exist_ok=True);
                    

                


                # for i in range(self.runConfig.parallel_processes):
                #     process = multiprocessing.Process(target=self.eval_training_data_batch_feedforward,args=(genomes,config,data_batches[i],i,lock));
                #     processes.append(process);
                # for process in processes:
                #     process.start();
                # for process in processes:
                #     process.join();
                # return;
            else:
                for datum in self.runConfig.training_data:
                    for genome_id, genome in genomes:
                        genome.fitness += self.eval_genome_feedforward(genome,config,trainingDatum=datum)



    def eval_genome_feedforward(self,genome,config,trainingDatum=None):
        return Genome_Executor.eval_genome_feedforward(genome,config,self.runConfig,self.game,trainingDatum=trainingDatum)



#Ok, so
#WHY does this exist, you may ask?
#This is pretty much entirely for multiprocessing reasons. These functions used to be part of the game_runner_neat class, but there ended up being a lot of pickling overhead, and - more importantly - process id assignment requires global variables. 
#Since global variables are hard and dumb, I use class variables and class methods instead. Basically the same thing, but still encapsulated.
#These functions were almost entirely cut&pasted from the above class, and the functions were aliased for backwards compatibility

#Class that handles any and all genome processing, packaged and globalized for easier interface with parallelism
class Genome_Executor:
    pnum = None;
    global_game = None;
    count = 0;
    generation = None;
    last_checkpoint_time = None;
    tr = None;
    iterations_between = 0;

    #TODO: Abstractify this using gameClass methods
    @classmethod
    def initProcess(cls,id_queue,gameClass):
        cls.pnum = id_queue.get();
        # from py_mario_bros.PythonSuperMario_master.source import tools
        # from py_mario_bros.PythonSuperMario_master.source import constants as c
        # if (cls.pnum == 0):
        #     c.GRAPHICS_SETTINGS = c.LOW;
        # else:
        #     c.GRAPHICS_SETTINGS = c.NONE;
        # # from py_mario_bros.PythonSuperMario_master.source.states.segment import Segment
        # cls.global_game = tools.Control(process_num=cls.pnum);
        # state_dict = {c.LEVEL: Segment()};
        # cls.global_game.setup_states(state_dict, c.LEVEL);
        # cls.global_game.state.startup(0,{c.LEVEL_NUM:None});
        cls.count = 0;
        cls.tr = tracker.SummaryTracker();
        cls.tr.diff();



    #process methods - iterate within
    @classmethod
    def eval_genome_batch_feedforward(cls,config,runnerConfig,game,genomes,gen=None):
        if gen is not None:
            if gen != cls.generation:
                cls.count = 0;
            cls.generation = gen;
        fitnesses = {genome_id:0 for genome_id,genome in genomes};
        for genome_id, genome in genomes:
            cls.count += 1;
            if cls.count % 100 == 0:
                time = datetime.now()
                print(f'Parallel Checkpoint - Process #{cls.pnum} at {time}' + ('' if cls.generation is None else f'; Count: {cls.count} evals completed this generation ({cls.generation})') + ('' if cls.last_checkpoint_time is None else f'; Eval Speed: {100/(time-cls.last_checkpoint_time).total_seconds()}'));
                cls.last_checkpoint_time = time;
            fitnesses[genome_id] += cls.eval_genome_feedforward(genome,config,runnerConfig,game);
        return fitnesses;

    @classmethod
    def eval_training_data_batch_feedforward(cls,config,runnerConfig,game,genomes,data,gen=None):
        if gen is not None:
            if gen != cls.generation:
                cls.count = 0;
            cls.generation = gen;
        count = 0;
        fitnesses = {genome_id:0 for genome_id,genome in genomes};
        for datum in data:
            for genome_id,genome in genomes:
                fitnesses[genome_id] += cls.eval_genome_feedforward(genome,config,runnerConfig,game,trainingDatum=datum);
                cls.count += 1;
                if cls.count % 100 == 0:
                    time = datetime.now()
                    print(f'Parallel Checkpoint - Process #{cls.pnum} at {time}' + ('' if cls.generation is None else f'; Count: {cls.count} evals completed this generation ({cls.generation})') + ('' if cls.last_checkpoint_time is None else f'; Eval Speed: {100/(time-cls.last_checkpoint_time).total_seconds()}'));
                    cls.last_checkpoint_time = time;
        return fitnesses;

    #map methods - iterate externally
    @classmethod
    def map_eval_genomes_feedforward(cls,config,runnerConfig,game,genomes,datum,gen=None):
        if gen is not None:
            if gen != cls.generation:
                cls.count = 0;
            cls.generation = gen;
        fitnesses = {genome_id:0 for genome_id,genome in genomes};
        for genome_id,genome in genomes:
                cls.count += 1;
                if cls.count % 100 == 0:
                    time = datetime.now()
                    print(f'Parallel Checkpoint - Process #{cls.pnum} at {time}' + ('' if cls.generation is None else f'; Count: {cls.count} evals completed this generation ({cls.generation})') + ('' if cls.last_checkpoint_time is None else f'; Eval Speed: {100/(time-cls.last_checkpoint_time).total_seconds()}'));
                    cls.last_checkpoint_time = time;
                
                fitnesses[genome_id] += cls.eval_genome_feedforward(genome,config,runnerConfig,game,trainingDatum=datum);
                
        return fitnesses;

    @classmethod
    def map_eval_genome_feedforward(cls,config,runnerConfig,game,genome,gen=None):
        if gen is not None:
            if gen != cls.generation:
                cls.count = 0;
            cls.generation = gen;
        cls.count += 1;
        if cls.count % 100 == 0:
            time = datetime.now()
            print(f'Parallel Checkpoint - Process #{cls.pnum} at {time}' + ('' if cls.generation is None else f'; Count: {cls.count} evals completed this generation ({cls.generation})') + ('' if cls.last_checkpoint_time is None else f'; Eval Speed: {100/(time-cls.last_checkpoint_time).total_seconds()}'));
            cls.last_checkpoint_time = time;
        return cls.eval_genome_feedforward(genome,config,runnerConfig,game);

    @classmethod
    def eval_genome_feedforward(cls,genome,config,runnerConfig,game,trainingDatum=None):
        #print('genome evaluation triggered');
        net = neat.nn.FeedForwardNetwork.create(genome,config);
        fitnesses = [];
        
        for trial in range(runnerConfig.numTrials):
            #startTime = time.time()
            #print('evaluating genome with id {0}, trial {1}'.format(genome.key,trial));
            fitness = 0;
            runningGame = game.start(runnerConfig,training_datum = trainingDatum, process_num = cls.pnum);
            if runnerConfig.fitness_collection_type != None and 'delta' in runnerConfig.fitness_collection_type:
                fitness -= runningGame.getFitnessScore();


            while (runningGame.isRunning()):
                #get the current data from the running game, as specified by the runnerConfig
                
                gameData = runningGame.getData();


                #print('input: {0}'.format(gameData));
                try:
                    gameInput = net.activate(gameData);
                except:
                    print('Error in activating net with data ', gameData, ' and mapped data ', runningGame.getMappedData());
                    print('Error body: ', sys.exc_info()[0]);
                    raise

                runningGame.processInput(gameInput);

                if (runnerConfig.fitness_collection_type != None and 'continuous' in runnerConfig.fitness_collection_type):
                    fitness += runningGame.getFitnessScore();

            fitness += runningGame.getFitnessScore();
            fitnesses.append(fitness);
            runningGame.close();
            #print(time.time()-startTime)

        fitness = runnerConfig.fitnessFromArray()(fitnesses);
        return fitness;
        #print(genome.fitness);
    
