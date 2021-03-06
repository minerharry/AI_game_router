from game_runner_neat import GameRunner 
from runnerConfiguration import RunnerConfig, IOData
from baseGame import EvalGame
from py_mario_bros.PythonSuperMario_master.smb_game import SMB1Game
from py_mario_bros.PythonSuperMario_master.source.states.segmentGenerator import SegmentGenerator,GenerationOptions
import os
import neat
import multiprocessing
try:
   import cPickle as pickle
except:
   import pickle
from py_mario_bros.PythonSuperMario_master.source import tools
from py_mario_bros.PythonSuperMario_master.source import constants as c
import run_states

steps_threshold = 800;


def task_obstruction_score(obstructions):
    return -obstructions[0] #- 0.4 * obstructions[1] - 0.1 * obstructions[2];

def getFitness(inputs):
    obstructions = inputs['task_obstructions'];
    return task_obstruction_score(obstructions) + inputs['player_state'] + inputs['task_reached']*50;

def getRunning(inputs):
    return (not(inputs['done']) and (not inputs['stillness_time'] > steps_threshold));




if __name__ == "__main__":
    import gc
    #gc.set_debug(gc.DEBUG_LEAK);
    #TODO: add training data manager class

    multiprocessing.freeze_support();


    run_state = run_states.CONTINUE;
    currentRun = 6;
    manual_continue_generation = 18;

    reRunGeneration = 50;
    reRunId = 88;

    





    configs = [
        GenerationOptions(num_blocks=0,ground_height=[7,9],valid_task_blocks=c.FLOOR,valid_start_blocks=c.FLOOR),
        GenerationOptions(num_blocks=[0,4],ground_height=[7,9],task_batch_size=[1,4]),
        GenerationOptions(num_blocks=[0,8],ground_height=[7,10],task_batch_size=[1,4]),
        GenerationOptions(num_blocks=[0,6],ground_height=[7,9],task_batch_size=[1,4],num_enemies={c.ENEMY_TYPE_GOOMBA:[0,1]}),
        ];

    training_data = [];
    if (run_state == run_states.NEW):
        inital_config = configs[0]
        training_data = SegmentGenerator.generateBatch(inital_config,4);
        os.makedirs(f"memories\\smb1Py\\",exist_ok=True);
        f = open(f'memories\\smb1Py\\run-{currentRun}-data','wb');
        pickle.dump(training_data,f);
        f.close();
    else:
        f = open(f'memories\\smb1Py\\run-{currentRun}-data','rb')
        training_data = pickle.load(f);
        f.close();

    add_data = True;
    additional_data_index = 1;

    if add_data:
        additional_config = configs[additional_data_index];
        training_data += SegmentGenerator.generateBatch(additional_config,50);


    runConfig = RunnerConfig(getFitness,getRunning,logging=True,parallel=True,gameName='smb1Py',returnData=['player_state',IOData('vel','array',array_size=[2]),IOData('task_position','array',array_size=[2]),IOData('pos','array',array_size=[2]),IOData('collision_grid','array',[15,15]),IOData('enemy_grid','array',[15,15]),IOData('box_grid','array',[15,15]),IOData('brick_grid','array',[15,15]),IOData('powerup_grid','array',[15,15])],num_trials=1,num_generations=None);
    runConfig.tile_scale = 2;
    runConfig.view_distance = 4 * runConfig.tile_scale - 1;
    runConfig.training_data = training_data;
    runConfig.task_obstruction_score = task_obstruction_score;
    runConfig.external_render = False;
    runConfig.parallel_processes = 4;
    runConfig.chunkFactor = 24;
    runConfig.saveFitness = True;

    runConfig.logPath = f'logs\\smb1Py\\run-{currentRun}-log.txt';
    runConfig.fitness_collection_type='delta';
    print(runConfig.gameName);


    game = EvalGame(SMB1Game);
    
#    print(game.initInputs);
    runner = GameRunner(game,runConfig);
    if (run_state == run_states.CONTINUE):
        winner = runner.continue_run('run_' + str(currentRun),manual_generation=manual_continue_generation);
        print('\nBest genome:\n{!s}'.format(winner));
    else:
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, 'config-pygame-smb1')
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
        if (run_state == run_states.NEW):
            winner = runner.run(config,'run_' + str(currentRun));
            print('\nBest genome:\n{!s}'.format(winner))
        if (run_state == run_states.RERUN):
            runner.replay_best(reRunGeneration,config,'run_' + str(currentRun),net=False,randomReRoll=True);
        if (run_state == run_states.RERUN_ALL):
            runner.replay_generation(reRunGeneration,'run_' + str(currentRun));
        if (run_state == run_states.RERUN_ID):
            runner.render_genome_by_id(reRunId,reRunGeneration,config,'run_' + str(currentRun),net=True);
