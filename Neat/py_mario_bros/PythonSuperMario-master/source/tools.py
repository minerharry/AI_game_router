__author__ = 'marble_xu'

import os
import pygame as pg
from abc import ABC, abstractmethod
from . import constants as c

keybinding = {
    'action':pg.K_s,
    'jump':pg.K_a,
    'left':pg.K_LEFT,
    'right':pg.K_RIGHT,
    'down':pg.K_DOWN,
    'save_state':pg.K_PAGEUP,
    'load_state':pg.K_PAGEDOWN,
}

class State():
    def __init__(self):
        self.start_time = 0.0
        self.current_time = 0.0
        self.done = False
        self.next = None
        self.persist = {}
    
    @abstractmethod
    def startup(self, current_time, persist):
        '''abstract method'''

    def cleanup(self):
        self.done = False
        return self.persist
    
    @abstractmethod
    def update(self, surface, keys, current_time):
        '''abstract method'''

class Control():
    def __init__(self):
        self.screen = pg.display.get_surface()
        self.done = False
        self.clock = pg.time.Clock()
        self.fps = 60
        self.current_time = 0.0
        manual_mode = True;
        if (manual_mode):
            self.keys = pg.key.get_pressed();
        else:
            self.keys = [];
        
        self.state_dict = {}
        self.state_name = None
        self.state = None
        self.fps_counter = 0;
        self.last_fps_time = pg.time.get_ticks();
        self.fps_cycle = 50;
    
    def setup_states(self, state_dict, start_state):
        self.state_dict = state_dict
        self.state_name = start_state
        self.state = self.state_dict[self.state_name]
    
    def update(self,auto_advance_state = True):
        if (self.fps_counter >= self.fps_cycle):
            self.fps_counter = 0;
            if (c.DISPLAY_FRAMERATE):
                 print("fps: " + str(1000 * self.fps_cycle/(self.current_time - self.last_fps_time)));
            self.last_fps_time = self.current_time;
        self.fps_counter += 1;

        self.current_time = pg.time.get_ticks()
        if self.state.done and auto_advance_state:
            self.flip_state()
        self.state.update(self.screen, self.keys, self.current_time)
    
    def flip_state(self):
        previous, self.state_name = self.state_name, self.state.next
        persist = self.state.cleanup()
        self.state = self.state_dict[self.state_name]
        self.state.startup(self.current_time, persist)

    def event_loop(self):
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.done = True
            elif event.type == pg.KEYDOWN:
                self.keys = pg.key.get_pressed()
            elif event.type == pg.KEYUP:
                self.keys = pg.key.get_pressed()
    
    def main(self):
        timer = 0;
        while not self.done:
            self.event_loop()
            self.update()
            if (timer <= 10):
                pg.display.update()
                
            #self.clock.tick(self.fps)

    def tick_inputs(self,named_inputs,show_game=False):
        keys = {};
        for name,val in named_inputs:
            keys[keybinding[name]] = val;
        self.keys = keys;
        self.update(auto_advance_state=False);
        if (show_game):
            pg.display.update();


    def get_game_data(self):
        return self.state.get_game_data;

    #only called during manual mode, so can be assumed that current state is a segment
    def load_segment(self,load_data):
        self.state.load(load_data);

    def accepts_player_input(self):
        return self.state.accepts_player_input();


    



def get_image(sheet, x, y, width, height, colorkey, scale):
        image = pg.Surface([width, height])
        rect = image.get_rect()

        image.blit(sheet, (0, 0), (x, y, width, height))
        image.set_colorkey(colorkey)
        image = pg.transform.scale(image,
                                   (int(rect.width*scale),
                                    int(rect.height*scale)))
        return image

def load_all_gfx(directory, colorkey=(255,0,255), accept=('.png', '.jpg', '.bmp', '.gif')):
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if ext.lower() in accept:
            img = pg.image.load(os.path.join(directory, pic))
            if img.get_alpha():
                img = img.convert_alpha()
            else:
                img = img.convert()
                img.set_colorkey(colorkey)
            graphics[name] = img
    return graphics
