from game_runner_neat import GameRunner 
from runnerConfiguration import RunnerConfig, IOData
from baseGame import EvalGame
from py_mario_bros.PythonSuperMario_master.smb_game import SMB1Game
from py_mario_bros.PythonSuperMario_master.source.states.segmentGenerator import SegmentGenerator,GenerationOptions
import os
import neat
try:
   import cPickle as pickle
except:
   import pickle
from py_mario_bros.PythonSuperMario_master.source import setup, tools
from py_mario_bros.PythonSuperMario_master.source import constants as c
from py_mario_bros.PythonSuperMario_master.source.states.segment import Segment



gameEnv = tools.Control()
state_dict = {c.LEVEL: Segment()}
gameEnv.setup_states(state_dict, c.LEVEL)
gameEnv.state.startup(0,{c.LEVEL_NUM:1});


#run_state = 'continue';
run_state = 'rerun';
#run_state = 'new';
currentRun = 3;
reRunGeneration = 20

if (False):
    f = open(f'memories\\smb1Py\\run-{currentRun}-data','rb')
    training_data = pickle.load(f);
    f.close();
    print(training_data[0].static_data);
    while (True):
        x=3;




steps_threshold = 1000;


def task_obstruction_score(obstructions):
    return -obstructions[0] - 0.4 * obstructions[1] - 0.1 * obstructions[2];

def getFitness(inputs):
    obstructions = inputs['task_obstructions'];
    return task_obstruction_score(obstructions);

def getRunning(inputs):
    return (not(inputs['done']) and (not inputs['stillness_time'] > steps_threshold));

training_data = [];
if (run_state == 'new'):
    inital_config = GenerationOptions(num_blocks=[0,5],ground_height=[0,0]);
    training_data = SegmentGenerator.generateBatch(inital_config,75);
    f = open(f'memories\\smb1Py\\run-{currentRun}-data','wb');
    pickle.dump(training_data,f);
    f.close();
else:
    f = open(f'memories\\smb1Py\\run-{currentRun}-data','rb')
    training_data = pickle.load(f);
    f.close();


runConfig = RunnerConfig(getFitness,getRunning,logging=True,parallel=False,gameName='smb1Py',returnData=['player_state',IOData('vel','array',array_size=[2]),IOData('task_position','array',array_size=[2]),IOData('pos','array',array_size=[2]),IOData('collision_grid','array',[15,15]),IOData('enemy_grid','array',[15,15]),IOData('box_grid','array',[15,15]),IOData('brick_grid','array',[15,15]),IOData('powerup_grid','array',[15,15])],num_trials=1,num_generations=None);
runConfig.tile_scale = 2;
runConfig.view_distance = 4 * runConfig.tile_scale - 1;
runConfig.training_data = training_data;
runConfig.task_obstruction_score = task_obstruction_score;
runConfig.external_render = False;

runConfig.logPath = f'logs\\smb1Py\\run-{currentRun}-log.txt';
runConfig.fitness_collection_type='continuous';
print(runConfig.gameName);


game = EvalGame(SMB1Game,game=gameEnv);
runner = GameRunner(game,runConfig);
if (run_state == 'continue'):
    winner = runner.continue_run('run_' + str(currentRun));
    print('\nBest genome:\n{!s}'.format(winner));
else:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-pygame-smb1')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    if (run_state == 'new'):
        winner = runner.run(config,'run_' + str(currentRun));
        print('\nBest genome:\n{!s}'.format(winner))
    if (run_state == 'rerun'):
        runner.replay_best(reRunGeneration,config,'run_' + str(currentRun),net=False);
    #runner.check_output_connections(10,config,'run_' + str(currentRun),3);
