from game_runner_neat import GameRunner 
from runnerConfiguration import RunnerConfig, IOData
from baseGame import EvalGame
from py_mario_bros.PythonSuperMario-master.smb_game import SMB1Game
import os
import neat
from py_mario_bros.PythonSuperMario-master import setup, tools
from py_mario_bros.PythonSuperMario-master import constants as c
from py_mario_bros.PythonSuperMario-master.states import segment

gameEnv = tools.Control()
state_dict = {c.LEVEL: segment.Segment()}
    game.setup_states(state_dict, c.MAIN_MENU)
    game.main()

game = EvalGame(NesPyGymGame,env=smb1Env);
continueRun = False;
continueRunRun = 8
newRun = True;
currentRun = 9;
reRun = False;
reRunGen = 8;
reRunRun = 1;
steps_threshold = 600;

smb1Env.setWindowName('SuperMarioBros-v0 run {0}'.format(continueRunRun if continueRun else (currentRun if newRun else reRunRun)));

def getFitness(inputs):
    return inputs['gym-reward'];

def getRunning(inputs):
    return (not(inputs['done']) and (not inputs['stillness_time'] > steps_threshold));


runConfig = NESGymRunnerConfig(getFitness,getRunning,parallel=False,gameName='gym_nes_smb1',returnData=['stage','status','world','x_pos','y_pos',IOData('enemy_type','array',array_size=[5]),IOData('enemy_x','array',array_size=[5]),IOData('enemy_y','array',array_size=[5]),IOData('blocks','array',array_size=[8,8]),IOData('enemies','array',array_size=(8,8)),'powerup_x','powerup_y'],num_trials=1,num_generations=None);
runConfig.logging = True;
runConfig.logPath = f'logs\\smb1\\run-{currentRun}-log.txt';
runConfig.playback_fps = 20;
runConfig.fitness_collection_type='continuous'
print(runConfig.gameName);

runner = GameRunner(game,runConfig);
if (continueRun):
    winner = runner.continue_run('run_' + str(continueRunRun));
    print('\nBest genome:\n{!s}'.format(winner));
else:
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-gym-nes-smb1')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    if (newRun):
        winner = runner.run(config,'run_' + str(currentRun));
        print('\nBest genome:\n{!s}'.format(winner))
    if (reRun):
        runner.replay_best(reRunGen,config,'run_' + str(reRunRun),net=True,randomReRoll=True);
