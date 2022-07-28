import json
from pathlib import Path
from tracemalloc import start
from game_runner_neat import GameRunner 
from runnerConfiguration import RunnerConfig, IOData
from baseGame import EvalGame
from games.smb1Py.py_mario_bros.PythonSuperMario_master.smb_game import SMB1Game
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segmentGenerator import SegmentGenerator,GenerationOptions
import os
import neat
import multiprocessing

from training_data import TrainingDataManager
try:
   import cPickle as pickle
except:
   import pickle
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import constants as c
import run_states as run_states

steps_threshold = 800;


def task_obstruction_score(obstructions):
    return -obstructions[0] #- 0.4 * obstructions[1] - 0.1 * obstructions[2];

def getFitness(inputs):
    obstructions = inputs['task_obstructions'];
    return task_obstruction_score(obstructions) + inputs['player_state'] + inputs['task_reached']*50;

def getRunning(inputs):
    return (not(inputs['done']) and (not inputs['stillness_time'] > steps_threshold));




if __name__ == "__main__":

    import sys
    sys.path.append('../core')
    
    NAME = "smb1Py";

    multiprocessing.freeze_support();


    run_state = run_states.RERUN_TOP;
    currentRun = 10;

    override_config = False;
    manual_continue_generation = None;
    diff_inputs = False;
    diff_outputs = False;
    output_map = None;
    prevData = ['player_state',
        IOData('vel','array',array_size=[2]),
        IOData('task_position_offset','array',array_size=[2]),
        IOData('pos','array',array_size=[2])];
    inputOptions = c.COLLISION_GRID;

    reRunGeneration = 1530;
    reRunPop = 15;
    reRunId = 88;
    customGenome = None;    

    
    ##TRAINING_DATA##
    
    set_data = True;
    add_data = False;
    complete_task_set = True;
    start_data_index = 6
    additional_data_indices = [2,4];

    configs = [
        GenerationOptions(num_blocks=0,ground_height=[7,9],valid_task_blocks=c.FLOOR,valid_start_blocks=c.FLOOR),
        GenerationOptions(num_blocks=0,ground_height=[7,9],valid_task_blocks=c.INNER,valid_start_blocks=c.FLOOR),
        GenerationOptions(num_blocks=[1,3],ground_height=[7,9],valid_task_blocks=c.INNER,valid_start_blocks=c.FLOOR),
        GenerationOptions(num_blocks=[0,4],ground_height=[7,9],task_batch_size=[1,4]),
        GenerationOptions(num_blocks=[0,8],ground_height=[7,10],task_batch_size=[1,4]),
        GenerationOptions(num_blocks=[0,6],ground_height=[7,9],task_batch_size=[1,4],num_enemies={c.ENEMY_TYPE_GOOMBA:[0,1]}),
        GenerationOptions(num_blocks=[0,8],ground_height=[7,10],task_batch_size=1000),
        ];
    
    tdManager = TrainingDataManager(NAME,currentRun);
    if (run_state == run_states.NEW or set_data):
        if not complete_task_set: 
            data = SegmentGenerator.generateBatch(configs[start_data_index],20);
            tdManager.set_data(data);
        else:
            id_sets = [];
            tdManager.clear_data();
            for i in range(100):
                data = SegmentGenerator.generate(configs[start_data_index],makeBatches=True);
                id_sets.append(tdManager.add_data(data));
                print(len(id_sets[-1]));
            id_store = Path("memories")/"smb1Py"/"run-10-taskarray-data";
            if os.path.exists(id_store):
                with open(id_store,'r') as f:
                    id_sets += json.load(f);
            with open(id_store,'w') as f:
                json.dump(id_sets,f);
    if add_data:
        for idx in additional_data_indices:
            tdManager.add_data(SegmentGenerator.generateBatch(configs[idx],20));

    print(f"Training Data Size: {len(tdManager.active_data)}");

    inputData = [
        'player_state',
        IOData('vel','array',array_size=[2]),
        IOData('task_position_offset','array',array_size=[2]),
        IOData('pos','array',array_size=[2])];
    config_suffix = "-nogrid"

    if inputOptions == c.FULL:
        inputData += [
            IOData('collision_grid','array',[15,15]),
            IOData('enemy_grid','array',[15,15]),
            IOData('box_grid','array',[15,15]),
            IOData('brick_grid','array',[15,15]),
            IOData('powerup_grid','array',[15,15])]
        config_suffix = "-full"
    if inputOptions == c.COLLISION_GRID:
        inputData.append(IOData('collision_grid','array',[15,15]))
        config_suffix = "-blockgrid"

    runConfig = RunnerConfig(
        getFitness,
        getRunning,
        logging=True,
        parallel=True,
        gameName=NAME,
        returnData=inputData,
        num_trials=1,
        num_generations=None,
        training_data=tdManager);

    runConfig.tile_scale = 2;
    runConfig.view_distance = 4 * runConfig.tile_scale - 1;
    runConfig.task_obstruction_score = task_obstruction_score;
    runConfig.external_render = False;
    runConfig.parallel_processes = 12;
    runConfig.chunkFactor = 48;
    runConfig.saveFitness = True;

    runConfig.logPath = f'logs\\smb1Py\\run-{currentRun}-log.txt';
    runConfig.fitness_collection_type='delta';



    game = EvalGame(SMB1Game);
    
    runner = GameRunner(game,runConfig);
    config_path = os.path.join(os.path.dirname(__file__), 'configs','config-pygame-smb1' + config_suffix);
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path);
    config_transfer = None;
    if override_config:
        config_transfer = (config, runConfig.get_input_transfer(prevData) if diff_inputs else None, output_transfer if diff_outputs else None)

    if (run_state == run_states.EVAL_CUSTOM):
        customGenome = neat.genome.DefaultGenome(0);
        customGenome.configure_new(config.genome_config);
        customGenome.add_connection(config.genome_config,-1,2,-1,True);
        customGenome.add_connection(config.genome_config,-1,3,1,True);
    if (run_state == run_states.CONTINUE):
        winner = runner.continue_run('run_' + str(currentRun),manual_generation=manual_continue_generation,manual_config_override=config_transfer);
        print('\nBest genome:\n{!s}'.format(winner));
    else:
        local_dir = os.path.dirname(__file__)

        if (run_state == run_states.NEW):
            winner = runner.run(config,'run_' + str(currentRun));
            print('\nBest genome:\n{!s}'.format(winner))
        if (run_state == run_states.RERUN):
            runner.replay_best(reRunGeneration,config,'run_' + str(currentRun),net=True,randomReRoll=True);
        if (run_state == run_states.RERUN_ALL):
            runner.replay_generation(reRunGeneration,'run_' + str(currentRun));
        if (run_state == run_states.RERUN_ID):
            runner.render_genome_by_id(reRunId,reRunGeneration,config,'run_' + str(currentRun),net=True);
        if (run_state == run_states.RERUN_TOP):
            runner.run_top_genomes(reRunGeneration,config,'run_' + str(currentRun),reRunPop,doFitness=True,randomReRoll=True);
        if (run_state == run_states.EVAL_CUSTOM):
            runner.render_custom_genome_object(customGenome,config,net=True)
