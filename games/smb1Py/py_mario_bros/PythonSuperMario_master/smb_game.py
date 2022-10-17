from baseGame import RunGame,EvalGame
from abc import abstractmethod

from runnerConfiguration import RunnerConfig
from .source import tools
from .source import constants as c
from .source.states.segment import Segment


empty_actions = dict(zip(['action','jump','left','right','down'],[False for i in range(5)]))
class SMB1Game(RunGame):

    @classmethod
    def initProcess(cls, pnum: int, parent_game: EvalGame):
        game = tools.Control(process_num=pnum);
        state_dict = {c.LEVEL: Segment()};
        game.setup_states(state_dict, c.LEVEL)
        game.state.startup(0,{c.LEVEL_NUM:1});
        parent_game.initInputs['game'] = game;

    
    def __init__(self,runnerConfig:RunnerConfig,**kwargs):
        self.runConfig = runnerConfig;
        datum = None
        if 'training_datum_id' in kwargs:
            datum = self.runConfig.training_data[kwargs['training_datum_id']];
        if 'game' not in kwargs:
            if 'GRAPHICS_SETTINGS' in kwargs:
                c.GRAPHICS_SETTINGS = kwargs['GRAPHICS_SETTINGS'];
            elif 'num_rendered_processes' in kwargs and 'process_num' in kwargs:
                if (kwargs['process_num']>=kwargs['num_rendered_processes']):
                    c.GRAPHICS_SETTINGS = c.NONE;
            if 'process_num' in kwargs:
                self.process_num = kwargs['process_num']
                self.game = tools.Control(process_num=self.process_num)
            else:
                self.game = tools.Control();
            state_dict = {c.LEVEL: Segment()}
            self.game.setup_states(state_dict, c.LEVEL)
            self.game.state.startup(0,{c.LEVEL_NUM:1},initial_state=datum);
            kwargs['game'] = self.game;
        else:
            self.game = kwargs["game"];
            self.game.load_segment(datum);
        self.min_obstructions = None;
        self.stillness_time = 0;
        self.annotations = kwargs['annotations'] if 'annotations' in kwargs else [];
        self.last_path = None;
        self.frame_counter = 0;
        self.frame_cycle = 4 if not hasattr(runnerConfig,"frame_cycle") else runnerConfig.frame_cycle;
        super().__init__(runnerConfig,**kwargs);


    def getOutputData(self):
        data = self.game.get_game_data(self.runConfig.view_distance,self.runConfig.tile_scale);
        obstruction_score = self.runConfig.task_obstruction_score(data['task_obstructions']);
        path_progress = data['task_path_remaining'];
        path_improved = False;
        if (self.last_path is None):
            self.last_path = path_progress;
            if path_progress is not None:
                path_improved = True;
        elif (path_progress is not None and path_progress < self.last_path):
            self.last_path = path_progress;
            path_improved = True;
            
        if (self.min_obstructions is None or obstruction_score < self.min_obstructions or path_improved):
            self.stillness_time = 0;
            self.min_obstructions = obstruction_score;
        else:
            self.stillness_time += 1;
        data['stillness_time'] = self.stillness_time;
        return data;
        

    def processInput(self, inputs):
        output = [key > 0.5 for key in inputs];
        named_actions = dict(zip(['action','jump','left','right','down'],output));

        self.frame_counter = (self.frame_counter + 1)%self.frame_cycle;

        self.game.tick_inputs(named_actions,show_game=self.frame_counter==0);
        while (self.isRunning() and not self.game.accepts_player_input()): #skip bad frames
            self.game.tick_inputs(empty_actions,show_game=True);

        if self.annotations:
            import pygame as pg
            surface = pg.display.get_surface();
            for ann in self.annotations:
                pg.draw.circle(surface,c.GREEN,ann,4);
            pg.display.update();
        
        

    def renderInput(self,inputs):
        output = [key > 0.5 for key in inputs];
        
        named_actions = dict(zip(['action','jump','left','right','down'],output));

        self.game.tick_inputs(named_actions,show_game=True);
        while (self.isRunning() and not self.game.accepts_player_input()):
            self.game.tick_inputs(empty_actions);
            #print('skipping bad frames...');

