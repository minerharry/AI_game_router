__author__ = 'marble_xu'

import pygame as pg
from .. import setup, tools
from .. import constants as c
from . import coin, stuff, powerup

def create_brick(brick_group, item, level):
    if c.COLOR in item:
        color = item[c.COLOR]
    else:
        color = c.COLOR_TYPE_ORANGE

    x, y, type = item['x'], item['y'], item['type']
    if type == c.TYPE_COIN:
        brick_group.add(Brick(x, y, type, 
                    color, level.coin_group))
    elif (type == c.TYPE_STAR or
        type == c.TYPE_FIREFLOWER or
        type == c.TYPE_LIFEMUSHROOM):
        brick_group.add(Brick(x, y, type,
                    color, level.powerup_group))
    else:
        if c.BRICK_NUM in item:
            create_brick_list(brick_group, item[c.BRICK_NUM], x, y, type,
                        color, item['direction'])
        else:
            brick_group.add(Brick(x, y, type, color))
            
            
def create_brick_list(brick_group, num, x, y, type, color, direction):
    ''' direction:horizontal, create brick from left to right, direction:vertical, create brick from up to bottom '''
    size = 16 * c.SIZE_MULTIPLIER; #43 # 16 * c.BRICK_SIZE_MULTIPLIER is 43
    tmp_x, tmp_y = x, y
    for i in range(num):
        if direction == c.VERTICAL:
            tmp_y = y + i * size
        else:
            tmp_x = x + i * size
        brick_group.add(Brick(tmp_x, tmp_y, type, color))
        
class Brick(stuff.Stuff):
    def __init__(self, x, y, type, color=c.ORANGE, group=None, name=c.MAP_BRICK):
        orange_rect = [(16, 0, 16, 16), (432, 0, 16, 16)]
        green_rect = [(208, 32, 16, 16), (48, 32, 16, 16)]
        if color == c.COLOR_TYPE_ORANGE:
            frame_rect = orange_rect
        else:
            frame_rect = green_rect
        stuff.Stuff.__init__(self, x, y, 'tile_set',
                        frame_rect, c.BRICK_SIZE_MULTIPLIER)
        if c.GRAPHICS_SETTINGS == c.LOW:
            for frame in self.frames:
                if frame is not None:
                    frame.fill(c.BRICK_PLACEHOLDER_COLOR);
        self.color = color;
        self.compressed = False;
        self.rest_height = y
        self.state = c.RESTING
        self.y_vel = 0
        self.gravity = 1.2
        self.type = type
        if self.type == c.TYPE_COIN:
            self.coin_num = 10
        else:
            self.coin_num = 0
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
        if c.GRAPHICS_SETTINGS >= c.MED:
            for image_rect in self.image_rect_list:
                self.frames.append(tools.get_image(setup.GFX[self.sheet_name], 
                        *image_rect, c.BLACK, self.scale))
        elif c.GRAPHICS_SETTINGS == c.LOW:
            sizes = [];
            for frame in self.image_rect_list:
                rect = (frame[2],frame[3])
                if (rect not in sizes):
                    sizes.append(rect);
                    frame = pg.Surface((rect[0]*c.SIZE_MULTIPLIER,rect[1]*c.SIZE_MULTIPLIER)).convert();
                    frame.fill(c.BRICK_PLACEHOLDER_COLOR);
                    self.frames.append(frame);
                else:
                    self.frames.append(None);
        if c.GRAPHICS_SETTINGS != c.NONE:
            self.image = self.frames[self.frame_index]
            self.image.get_rect().x = self.rect.x;
            self.image.get_rect().bottom = self.rect.bottom;
        if self.type == c.TYPE_COIN:
            self.group = level.coin_group;
        else:
            self.group = level.powerup_group;
        if self.group_ids is not None:
            self.add([level.get_group_by_id(id) for id in self.group_ids if level.get_group_by_id(id) is not None]);
            self.group_ids = None;

        
    
    
    def update(self):
        if self.state == c.BUMPED:
            self.bumped()
    
    def bumped(self):
        self.rect.y += self.y_vel
        self.y_vel += self.gravity
        
        if self.rect.y >= self.rest_height:
            self.rect.y = self.rest_height
            if self.type == c.TYPE_COIN:
                if self.coin_num > 0:
                    self.state = c.RESTING
                else:
                    self.state = c.OPENED
            elif self.type == c.TYPE_STAR:
                self.state = c.OPENED
                self.group.add(powerup.Star(self.rect.centerx, self.rest_height))
            elif self.type == c.TYPE_FIREFLOWER:
                self.state = c.OPENED
                self.group.add(powerup.FireFlower(self.rect.centerx, self.rest_height))
            elif self.type == c.TYPE_LIFEMUSHROOM:
                self.state = c.OPENED
                self.group.add(powerup.LifeMushroom(self.rect.centerx, self.rest_height))
            else:
                self.state = c.RESTING
        
    def start_bump(self, score_group):
        self.y_vel -= 7
        
        if self.type == c.TYPE_COIN:
            if self.coin_num > 0:
                #self.group.add(coin.Coin(self.rect.centerx, self.rect.y, score_group))
                self.coin_num -= 1
                if self.coin_num == 0 and c.GRAPHICS_SETTINGS == c.HIGH:
                    self.frame_index = 1
                    self.image = self.frames[self.frame_index]
        elif (self.type == c.TYPE_STAR or 
            self.type == c.TYPE_FIREFLOWER or 
            self.type == c.TYPE_LIFEMUSHROOM) and c.GRAPHICS_SETTINGS == c.HIGH:
                self.frame_index = 1
                self.image = self.frames[self.frame_index]
        
        self.state = c.BUMPED
    
    def change_to_piece(self, group):
        arg_list = [(self.rect.x, self.rect.y - (self.rect.height/2), -2, -12),
                    (self.rect.right, self.rect.y - (self.rect.height/2), 2, -12),
                    (self.rect.x, self.rect.y, -2, -6),
                    (self.rect.right, self.rect.y, 2, -6)]
        
        for arg in arg_list:
            group.add(BrickPiece(*arg))
        self.kill()
        
class BrickPiece(stuff.Stuff):
    def __init__(self, x, y, x_vel, y_vel):
        stuff.Stuff.__init__(self, x, y, 'tile_set',
            [(68, 20, 8, 8)], c.BRICK_SIZE_MULTIPLIER)
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.gravity = .8

    
    def update(self, *args):
        self.rect.x += self.x_vel
        self.rect.y += self.y_vel
        self.y_vel += self.gravity
        if self.rect.y > c.SCREEN_HEIGHT:
            self.kill()
    
