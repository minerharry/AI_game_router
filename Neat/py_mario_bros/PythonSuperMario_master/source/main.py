__author__ = 'marble_xu'

import pygame as pg
from . import setup, tools
from . import constants as c
from .states import main_menu, load_screen, level, segment

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
        game.state.startup(0,{c.LEVEL_NUM:1});
    game.main()
