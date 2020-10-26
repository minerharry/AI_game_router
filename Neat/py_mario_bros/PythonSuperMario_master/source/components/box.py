__author__ = 'marble_xu'

import pygame as pg
from .. import setup, tools
from .. import constants as c
from . import coin, powerup

class Box(pg.sprite.Sprite):
    def __init__(self, x, y, type, group=None, name=c.MAP_BOX):
        pg.sprite.Sprite.__init__(self)
        
        self.frames = []
        self.frame_index = 0
        self.frame_rect_list = [(384, 0, 16, 16), (400, 0, 16, 16), 
            (416, 0, 16, 16), (400, 0, 16, 16), (432, 0, 16, 16)]
        self.load_frames()
        if c.GRAPHICS_SETTINGS != c.NONE:
            self.image = self.frames[self.frame_index]
            self.rect = self.image.get_rect()
            self.rect.x = x
            self.rect.y = y
        else:
            self.rect = pg.Rect(x,y,self.frame_rect_list[self.frame_index][2],self.frame_rect_list[self.frame_index][3]);

        self.rest_height = y
        self.animation_timer = 0
        self.first_half = True   # First half of animation cycle
        self.state = c.RESTING
        self.y_vel = 0
        self.gravity = 1.2
        self.type = type
        self.group = group
        self.name = name
        self.group_ids = None;

    #removes all of the unneeded variables that remain constant (removes unpickleable objects)
    def compress(self,level):
        self.frames = [];
        self.image = None;
        self.group = None;
        self.group_ids = [level.get_group_id(group) for group in self._Sprite__g if level.get_group_id(group) is not None];
        self._Sprite__g = {};


    #adds back all of the unneeded variables that remain constant (adds back unpickleable objects)
    def decompress(self,level):
        self.load_frames();
        if c.GRAPHICS_SETTINGS != c.NONE:
            self.image = self.frames[self.frame_index]
            self.image.get_rect().x = self.rect.x;
            self.image.get_rect().bottom = self.rect.bottom;
        if (self.type == c.TYPE_COIN):
            self.group = level.coin_group;
        else:
            self.group = level.powerup_group;
        if self.group_ids is not None:
            self.add([level.get_group_by_id(id) for id in self.group_ids if level.get_group_by_id(id) is not None]);
            self.group_ids = None;


        
    def load_frames(self):
        sheet = setup.GFX['tile_set']
        if (c.GRAPHICS_SETTINGS >= c.MED):
            for frame_rect in self.frame_rect_list:
                self.frames.append(tools.get_image(sheet, *frame_rect, 
                                c.BLACK, c.BRICK_SIZE_MULTIPLIER))
        elif (c.GRAPHICS_SETTINGS == c.LOW):
            sizes = [];
            for frame in self.frame_rect_list:
                rect = (frame[2],frame[3])
                if (rect not in sizes):
                    sizes.append(rect);
                    frame = pg.Surface((rect[0]*c.SIZE_MULTIPLIER,rect[1]*c.SIZE_MULTIPLIER)).convert();
                    frame.fill(c.BOX_PLACEHOLDER_COLOR)
                    self.frames.append(frame);
                else:
                    self.frames.append(None);

    
    def update(self, game_info):
        self.current_time = game_info[c.CURRENT_TIME]
        if self.state == c.RESTING:
            self.resting()
        elif self.state == c.BUMPED:
            self.bumped()

    def resting(self):
        time_list = [375, 125, 125, 125]
        if (self.current_time - self.animation_timer) > time_list[self.frame_index] and c.GRAPHICS_SETTINGS == c.HIGH:
            self.frame_index += 1
            if self.frame_index == 4:
                self.frame_index = 0
            self.animation_timer = self.current_time

        if c.GRAPHICS_SETTINGS != c.NONE:
            self.image = self.frames[self.frame_index]
    
    def bumped(self):
        self.rect.y += self.y_vel
        self.y_vel += self.gravity
        
        if self.rect.y > self.rest_height + 5:
            self.rect.y = self.rest_height
            self.state = c.OPENED
            if self.type == c.TYPE_MUSHROOM:
                self.group.add(powerup.Mushroom(self.rect.centerx, self.rect.y))
            elif self.type == c.TYPE_FIREFLOWER:
                self.group.add(powerup.FireFlower(self.rect.centerx, self.rect.y))
            elif self.type == c.TYPE_LIFEMUSHROOM:
                self.group.add(powerup.LifeMushroom(self.rect.centerx, self.rect.y))
        if c.GRAPHICS_SETTINGS == c.HIGH:
            self.frame_index = 4
            self.image = self.frames[self.frame_index]
    
    def start_bump(self, score_group):
        self.y_vel = -6
        self.state = c.BUMPED
        
        #if self.type == c.TYPE_COIN:
            #self.group.add(coin.Coin(self.rect.centerx, self.rect.y, score_group))
