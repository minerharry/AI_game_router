__author__ = 'marble_xu'

import pygame as pg
from . import tools
from . import constants as c
from .states import main_menu, load_screen, level, segment
from .states.segmentGenerator import SegmentGenerator,GenerationOptions

def main():
    game = tools.Control()
    if c.GRAPHICS_SETTINGS >= c.MED:
        state_dict = {c.MAIN_MENU: main_menu.Menu(),
                      c.LOAD_SCREEN: load_screen.LoadScreen(),
                      c.LEVEL: segment.Segment(),
                      c.GAME_OVER: load_screen.GameOver(),
                    c.TIME_OUT: load_screen.TimeOut()}
        game.setup_states(state_dict, c.MAIN_MENU)
    else:
        state_dict = {c.LEVEL: segment.Segment()}
        game.setup_states(state_dict,c.LEVEL);
        inital_config = GenerationOptions(num_blocks=[0,5],ground_height=[7,10],num_enemies={c.ENEMY_TYPE_GOOMBA:[0,1]});
        
        #training_datum = SegmentGenerator.generate(inital_config)[0];

        game.state.startup(0,{c.LEVEL_NUM:1});#,initial_state=training_datum);
    game.main()
