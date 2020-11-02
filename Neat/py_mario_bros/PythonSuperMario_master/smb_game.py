from baseGame import RunGame
from abc import abstractmethod
from .source import tools
from .source import constants as c


empty_actions = dict(zip(['action','jump','left','right','down'],[False for i in range(5)]))
class SMB1Game(RunGame):
    
    def __init__(self,runnerConfig,kwargs):
        self.steps = 0;
        self.runConfig = runnerConfig;
        if 'game' not in kwargs: #initial setup
            from .source.states.segment import Segment
            self.process_num = kwargs['process_num']
            self.game = tools.Control(process_num=self.process_num)
            state_dict = {c.LEVEL: Segment()}
            self.game.setup_states(state_dict, c.LEVEL)
            self.game.state.startup(0,{c.LEVEL_NUM:1});
            kwargs['game'] = self.game;
        else:
            self.game = kwargs["game"];
        self.game.load_segment(kwargs['training_datum']);
        self.min_obstructions = None;
        self.stillness_time = 0;
        


    def getOutputData(self):
        data = self.game.get_game_data(self.runConfig);
        #print(data);
        obstruction_score = self.runConfig.task_obstruction_score(data['task_obstructions'])
        if (self.min_obstructions is None or obstruction_score < self.min_obstructions):
            self.stillness_time = 0;
            self.min_obstructions = obstruction_score;
        else:
            self.stillness_time += 1;
        data['stillness_time'] = self.stillness_time;
        return data;
        

    def processInput(self, inputs):
        #print('input processed')
        output = [key > 0 for key in inputs];
        named_actions = dict(zip(['action','jump','left','right','down'],output));
        self.game.tick_inputs(named_actions);

        while (self.isRunning() and not self.game.accepts_player_input()):
            self.game.tick_inputs(empty_actions);
            #print('skipping bad frames...');
        #print(game.don)
        

    def renderInput(self,inputs):
        #print('input rendered')
        output = [key > 0 for key in inputs];
        named_actions = dict(zip(['action','jump','left','right','down'],output));

        self.game.tick_inputs(named_actions,show_game=True);
        while (self.isRunning() and not self.game.accepts_player_input()):
            self.game.tick_inputs(empty_actions);
            #print('skipping bad frames...');
        #print(game.don);

    def close(self):
        #does nothing unless game needs it to
        return;
