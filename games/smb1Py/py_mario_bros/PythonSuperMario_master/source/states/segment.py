__author__ = 'minerharry'

from copy import copy
from email.headerregistry import Group
import os
import json
import math
from typing import Any, Tuple
import pygame as pg

from just_in_time_dict import JITDict
try:
   import cPickle as pickle
except:
   import pickle
from .. import setup, tools
from .. import constants as c
from ..components import info, stuff, player, brick, box, enemy, powerup, coin

TASK_PATH_CHANGE_THRESHOLD = 0.8*c.TILE_SIZE;

class SegmentState:
    def __init__(self, dynamic_data, static_data, task:None|tuple[float,float] = None, task_bounds=None, task_path:None|list[tuple[float,float]]=None, file_path = None, raw_data = None):
        self.last_grid = None;
        self.raw_data = raw_data;
        if (file_path is not None):
            with open(file_path,'rb') as f:
                self.raw_data = pickle.load(f);


        if self.raw_data is not None:
            self.static_data = self.raw_data['static']; #otherwise known as the map data
            self.dynamic_data =self.raw_data['dynamic'];
            self.task = self.raw_data['task'];
            self.task_bounds = self.raw_data['task_bounds'];
            self.task_path = self.raw_data['task_path'] if 'task_path' in self.raw_data else None;
        else:
            self.static_data = static_data;
            self.dynamic_data = dynamic_data;
            self.task = task;
            self.task_bounds = task_bounds; #left,right,top,bottom
            self.task_path = task_path;
            if self.task_path is not None:
                self.task = self.task_path[0];

    #not particularly used
    def save_file(self,file_path):
        with open(file_path,'wb') as f:
            pickle.dump({'static':self.static_data,'dynamic':self.dynamic_data,'task':self.task,'task_path':self.task_path,'task_bounds':self.task_bounds},f);

    def equal_static_data(self,other_data):
        return json.dumps(other_data) == json.dumps(self.static_data); #using json dumps to ensure that only the dict values are compared and not class sources; that way, if static data gets treated as a class for generation purposes, it can still work exactly the same

    def __copy__(self):
        return self.__deepcopy__();

    def __deepcopy__(self,memo=None):
        return SegmentState(None,None,raw_data=pickle.loads(pickle.dumps({'static':self.static_data,'dynamic':self.dynamic_data,'task':self.task,'task_path':self.task_path,'task_bounds':self.task_bounds})));


class Segment(tools.State):
#TODO: add back a level timer
    def __init__(self):
        tools.State.__init__(self)
        self.player = None

    def accepts_player_input(self):
        return self.player.accepts_input();

    def startup(self, current_time, persist, initial_state = None):
        self.loaded_segment = None;
        self.last_load = False; #whether the load state button was held last frame
        self.last_save = False; #whether the save state button was held last frame
        self.saved_state = None;
        self.last_game_data = None;
        #self.game_info = persist
        #self.persist = self.game_info
        #self.game_info[c.CURRENT_TIME] = current_time
        self.current_time = current_time;
        self.death_timer = 0
        self.castle_timer = 0
        self.bg_image = False;
        self.grid_rects = None;
        self.task_bounds = None;
        self.task:None|tuple[float,float] = None;
        self.task_reached = 0;
        self.task_path:None|list[tuple[float,float]] = None;
        self.blank = False;


        #self.moving_score_list = []
        #self.overhead_info = info.Info(self.game_info, c.LEVEL)
        if initial_state is None:
            if persist[c.LEVEL_NUM] is None:
                self.blank = True;
                self.map_data = None;
                self.blank_groups();
            else:
                self.level_num = persist[c.LEVEL_NUM];
                self.load_map()
                self.setup_maps()
                self.setup_background() #look into how to remove
                self.ground_group = self.setup_collide(c.MAP_GROUND)
                self.step_group = self.setup_collide(c.MAP_STEP)
                self.setup_pipe()
                self.setup_slider()
            #self.setup_static_coin()
                self.setup_brick_and_box()
                self.setup_player()
                self.setup_sprite_groups()
                self.setup_enemies()
                self.setup_checkpoints()
                self.setup_flagpole()

        else:
            self.map_data = None;
            self.blank_groups();
            self.load(initial_state);

        self.setup_group_map();
        

    def blank_groups(self):
        self.shell_group = self.dying_group = self.player_group = self.ground_step_pipe_group = self.powerup_group = self.ground_group = self.box_group = self.brick_group = self.brickpiece_group = self.checkpoint_group = self.coin_group = self.enemy_group = self.flagpole_group = pg.sprite.Group();

    def setup_group_map(self):
        self.group_map = {self.shell_group: c.SHELL_GROUP, self.dying_group: c.DYING_GROUP, self.player_group: c.PLAYER_GROUP, self.ground_step_pipe_group: c.GROUND_STEP_PIPE_GROUP, self.powerup_group: c.POWERUP_GROUP, self.ground_group: c.GROUND_GROUP, self.box_group: c.BOX_GROUP, self.brick_group: c.BRICK_GROUP, self.brickpiece_group: c.BRICKPIECE_GROUP, self.checkpoint_group: c.CHECKPOINT_GROUP, self.coin_group: c.COIN_GROUP, self.enemy_group: c.ENEMY_GROUP, self.flagpole_group: c.FLAGPOLE_GROUP}
        self.group_id_map = {v: k for k, v in self.group_map.items()}

    def load_map(self):
        map_file = 'level_' + str(self.level_num) + '.json'
        file_path = os.path.join('games\\smb1Py\\py_mario_bros\\PythonSuperMario_master','source', 'data', 'maps', map_file)
        f = open(file_path)
        self.map_data = json.load(f)
        f.close()

    def load_map_data(self,data,load_enemies=False):
        self.map_data = data;
        #print(data);
        self.setup_maps()
        self.setup_background()
        self.ground_group = self.setup_collide(c.MAP_GROUND)
        self.step_group = self.setup_collide(c.MAP_STEP)
        self.setup_pipe()
        if load_enemies:
            self.setup_enemies();

    def save_internal_state(self):
        self.saved_state = pickle.dumps(self.save_state());

    def load_internal_state(self):
        if (self.saved_state is None):
            print("Unable to load state: no state saved");
            return;
        self.load(pickle.loads(self.saved_state));

    #saves state within a level
    def save_state(self):
        
        dynamic_data = {'done':self.done,'start_x': self.start_x, 'end_x':self.end_x,'shells':self.shell_group,'time':self.current_time, 'enemy_group_list':self.enemy_group_list,'player':self.player_group, 'viewport_pos':[self.viewport.rect.x,self.viewport.rect.y],'flagpole':self.flagpole_group,'bricks':self.brick_group,'boxes':self.box_group,'powerups':self.powerup_group,'sliders':self.slider_group};
        self.compress_dynamics();
        #print(dir(self.box_group.sprites()[0]));
        dynamic_data = pickle.loads(pickle.dumps(dynamic_data));
        self.decompress_dynamics();

        static_data = self.map_data;
        return SegmentState(dynamic_data,static_data);

    def compress_dynamics(self):
        [[item.compress(self) for item in group] for group in self.enemy_group_list];
        [item.compress(self) for item in self.player_group];
        [item.compress(self) for item in self.shell_group];
        [item.compress(self) for item in self.flagpole_group];
        [item.compress(self) for item in self.box_group];
        [item.compress(self) for item in self.powerup_group];
        [item.compress(self) for item in self.slider_group];
        [item.compress(self) for item in self.brick_group];

    def decompress_dynamics(self):
        
        [[item.decompress(self) for item in group] for group in self.enemy_group_list];
        [[item.add(group) for item in group] for group in self.enemy_group_list];
        [item.decompress(self) for item in self.shell_group];
        [item.decompress(self) for item in self.player_group];
        [item.decompress(self) for item in self.flagpole_group];
        [item.decompress(self) for item in self.box_group];
        [item.decompress(self) for item in self.powerup_group];
        [item.decompress(self) for item in self.slider_group];
        [item.decompress(self) for item in self.brick_group];


    def load_dynamic(self,data):
        if data is None:
           data = {};
        if self.enemy_group is not None:
            [sprite.kill() for sprite in self.enemy_group];
        self.enemy_group = pg.sprite.Group();
        self.coin_group = pg.sprite.Group();
        self.dying_group = pg.sprite.Group();
        self.setup_checkpoints();
        if 'done' in data:
            self.done = data['done'];
        else:
            self.done = False;
        if 'time' in data:
            self.current_time = data['time'];
        else:
            self.current_time = 0;
        do_viewport = False;
        if 'viewport_pos' in data:
            self.viewport.x = data['viewport_pos'][0];
            self.viewport.y = data['viewport_pos'][1];
        else:
            do_viewport = True;
                    
        if 'player' in data:
            self.player = data['player'].sprites()[0];
            self.player_x = self.player.rect.centerx;
            self.player_y = self.player.rect.bottom;
            self.player_group = data['player'];
        else:
            self.setup_player(do_viewport=do_viewport);
            if self.player not in self.player_group:
                self.player_group = pg.sprite.Group(self.player);           
        if 'shells' in data:
            self.shell_group = data['shells'];
        else:
            self.shell_group = pg.sprite.Group()
        if 'flagpole' in data:
            self.flagpole_group = data['flagpole'];
        else:
            self.setup_flagpole();
        if 'bricks' in data:
            self.brick_group = data['bricks'];
        else:
            self.setup_brick();
        if 'powerups' in data:
            self.powerup_group = data['powerups'];
        else:
            self.powerup_group = pg.sprite.Group();
        if 'boxes' in data:
            self.box_group = data['boxes'];
        else:
            self.setup_box();
        
        if 'sliders' in data:
            self.slider_group = data['sliders'];
        else:
            self.setup_slider()
        if 'enemy_group_list' in data:
            self.enemy_group_list = data['enemy_group_list'];
        else:
            self.setup_enemies();
        if 'start_x' in data:
            self.start_x = data['start_x'];
            self.end_x = data['end_x'];
        self.ground_step_pipe_group = pg.sprite.Group(self.ground_group,self.pipe_group, self.step_group, self.slider_group)
        self.setup_group_map();
        self.decompress_dynamics();


    def load(self,data:SegmentState):
        self.loaded_segment = data;
        self.task = copy(data.task);
        self.task_bounds = copy(data.task_bounds);
        self.task_reached = 0;
        if hasattr(data,'task_path'):
            self.task_path = copy(data.task_path);
        if self.task_path is not None and (self.task != self.task_path[0]):
            print(f"WARNING: initial task {data.task} and path {data.task_path} do not match, overwriting task");
            self.task = self.task_path[0];

        self.death_timer = 0; #TODO: Save & reload these timer values instead of resetting them
        self.castle_timer = 0;
        if (self.map_data is None or not data.equal_static_data(self.map_data)):
            self.load_map_data(data.static_data);
        self.load_dynamic(data.dynamic_data);

        self.last_game_data = None;




        
    def get_group_by_id(self,group_id):
        return self.group_id_map[group_id] if group_id in self.group_id_map.keys() else None;

    def get_group_id(self,group):
        if (group in self.group_map.keys()):
            return self.group_map[group] 
        else:
            return None;

       
    def setup_background(self):
        if (c.MAP_IMAGE in self.map_data):
            img_name = self.map_data[c.MAP_IMAGE]
            self.background = setup.get_GFX()[img_name]
            self.bg_rect = self.background.get_rect()
            self.background = pg.transform.scale(self.background, 
                                        (int(self.bg_rect.width*c.BACKGROUND_MULTIPLER),
                                        int(self.bg_rect.height*c.BACKGROUND_MULTIPLER)))
            self.bg_rect = self.background.get_rect()
            self.level = pg.Surface((self.bg_rect.w, self.bg_rect.h)).convert()
            self.viewport = pg.Rect((0,0),c.SCREEN_SIZE);
            self.viewport.bottom = self.bg_rect.bottom;
            self.bg_image = True;
        else:
            self.bg_image = False;
            rect = self.map_bounds;
            self.background = pg.Surface((rect[1]-rect[0],rect[3]-rect[2])).convert();
            self.background.fill(c.SKY_BLUE)
            self.level = pg.Surface((rect[1]-rect[0],rect[3]-rect[2])).convert();
            self.viewport = pg.Rect((0,0),c.SCREEN_SIZE);
            self.viewport.bottom = rect[3];


    def setup_maps(self):
        self.map_list = []

        if c.MAP_MAPS in self.map_data:
            for data in self.map_data[c.MAP_MAPS]:
                self.map_list.append((data[c.MAP_BOUNDS],data[c.MAP_START]))
            self.map_bounds, player_start = self.map_list[0]
            self.player_x = player_start[0];
            self.player_y = player_start[1]; 
            
        else:
            self.map_bounds:Tuple[int,int,int,int] = (0,self.bg_rect.w,0,self.bg_rect.h);
            self.player_x = 110
            self.player_y = c.GROUND_HEIGHT
        
    def change_map(self, index, type):
        self.map_bounds, player_start = self.map_list[index]
        self.player_x = player_start[0];
        self.player_y = player_start[1]; 
        self.viewport.x = self.player_x - c.SCREEN_WIDTH//3;
        self.viewport.bottom = self.player_y + c.SCREEN_HEIGHT//3;
        if self.viewport.right > self.map_bounds[1]:
            self.viewport.right = self.map_bounds[1];
        if self.viewport.left < self.map_bounds[0]:
            self.viewport.left = self.map_bounds[0];
        if self.viewport.bottom > self.map_bounds[3]:
            self.viewport.bottom = self.map_bounds[3];
        if self.viewport.top < self.map_bounds[2]:
            self.viewport.top = self.map_bounds[2];
        



        if type == c.CHECKPOINT_TYPE_MAP:
            self.player.rect.centerx = self.player_x
            self.player.rect.bottom = self.player_y
            self.player.update_hitbox();
            self.player.state = c.STAND
        elif type == c.CHECKPOINT_TYPE_PIPE_UP:
            self.player.rect.centerx = self.player_x
            self.player.rect.bottom = c.GROUND_HEIGHT
            self.player.update_hitbox();
            self.player.state = c.UP_OUT_PIPE
            self.player.up_pipe_y = self.player_y
            
    def setup_collide(self, name):
        group = pg.sprite.Group()
        if name in self.map_data:
            for data in self.map_data[name]:
                group.add(stuff.Collider(data['x'], data['y'], 
                        data['width'], data['height'], name))
        return group

    def setup_pipe(self):
        self.pipe_group = pg.sprite.Group()
        if c.MAP_PIPE in self.map_data:
            for data in self.map_data[c.MAP_PIPE]:
                self.pipe_group.add(stuff.Pipe(data['x'], data['y'],
                    data['width'], data['height'], data['type']))

    def setup_slider(self):
        self.slider_group = pg.sprite.Group()
        if c.MAP_SLIDER in self.map_data:
            for data in self.map_data[c.MAP_SLIDER]:
                if c.VELOCITY in data:
                    vel = data[c.VELOCITY]
                else:
                    vel = 1
                self.slider_group.add(stuff.Slider(data['x'], data['y'], data['num'],
                    data['direction'], data['range_start'], data['range_end'], vel))

    def setup_static_coin(self):
        self.static_coin_group = pg.sprite.Group()
        if c.MAP_COIN in self.map_data:
            for data in self.map_data[c.MAP_COIN]:
                self.static_coin_group.add(coin.StaticCoin(data['x'], data['y']))

    def setup_brick_and_box(self):
        self.coin_group = pg.sprite.Group()
        self.powerup_group = pg.sprite.Group()
        self.setup_brick();
        self.setup_box();
        
        
    def setup_brick(self):
        self.brick_group = pg.sprite.Group()
        self.brickpiece_group = pg.sprite.Group()

        if c.MAP_BRICK in self.map_data:
            for data in self.map_data[c.MAP_BRICK]:
                brick.create_brick(self.brick_group, data, self)

    def setup_box(self):
        self.box_group = pg.sprite.Group()
        if c.MAP_BOX in self.map_data:
            for data in self.map_data[c.MAP_BOX]:
                if data['type'] == c.TYPE_COIN:
                    self.box_group.add(box.Box(data['x'], data['y'], data['type'], self.coin_group))
                else:
                    self.box_group.add(box.Box(data['x'], data['y'], data['type'], self.powerup_group))
    
    def setup_player(self,do_viewport = True):
        if self.player is None:
            self.player = player.Player(c.PLAYER_MARIO)
        else:
            self.player.restart()
        self.player.rect.centerx = self.player_x
        self.player.rect.bottom = self.player_y
        self.player.update_hitbox();
        if c.DEBUG:
            self.player.rect.x = c.DEBUG_START_X
            self.player.rect.bottom = c.DEBUG_START_y
        if (do_viewport):
            self.viewport.x = self.player.rect.x - c.SCREEN_WIDTH//3;
            self.viewport.bottom = self.player.rect.bottom + c.SCREEN_HEIGHT//3;
            if self.viewport.right > self.map_bounds[1]:
                self.viewport.right = self.map_bounds[1];
            if self.viewport.left < self.map_bounds[0]:
                self.viewport.left = self.map_bounds[0];
            if self.viewport.bottom > self.map_bounds[3]:
                self.viewport.bottom = self.map_bounds[3];
            if self.viewport.top < self.map_bounds[2]:
                self.viewport.top = self.map_bounds[2];
            

    def setup_enemies(self):
        self.enemy_group_list = []
        if c.MAP_ENEMY not in self.map_data:
            return
        for index,data in self.map_data[c.MAP_ENEMY].items():
            group = pg.sprite.Group()
            for item in data:
                group.add(enemy.create_enemy(item))
            self.enemy_group_list.append(group)

            if int(index) == -1:
                self.enemy_group.add(group);
                
            
    def setup_checkpoints(self):
        self.checkpoint_group = pg.sprite.Group()
        if c.MAP_CHECKPOINT not in self.map_data:
            return;
        for data in self.map_data[c.MAP_CHECKPOINT]:
            if c.ENEMY_GROUPID in data:
                enemy_groupid = data[c.ENEMY_GROUPID]
            else:
                enemy_groupid = 0
            if c.MAP_INDEX in data:
                map_index = data[c.MAP_INDEX]
            else:
                map_index = 0
            self.checkpoint_group.add(stuff.Checkpoint(data['x'], data['y'], data['width'], 
                data['height'], data['type'], enemy_groupid, map_index))
    
    def setup_flagpole(self):
        self.flagpole_group = pg.sprite.Group()
        if c.MAP_FLAGPOLE in self.map_data:
            for data in self.map_data[c.MAP_FLAGPOLE]:
                if data['type'] == c.FLAGPOLE_TYPE_FLAG:
                    sprite = stuff.Flag(data['x'], data['y'])
                    self.flag = sprite
                elif data['type'] == c.FLAGPOLE_TYPE_POLE:
                    sprite = stuff.Pole(data['x'], data['y'])
                else:
                    sprite = stuff.PoleTop(data['x'], data['y'])
                self.flagpole_group.add(sprite)
        
        
    def setup_sprite_groups(self):
        self.dying_group = pg.sprite.Group()
        self.enemy_group = pg.sprite.Group()
        self.shell_group = pg.sprite.Group()
        
        self.ground_step_pipe_group = pg.sprite.Group(self.ground_group,
                        self.pipe_group, self.step_group, self.slider_group)
        self.player_group = pg.sprite.Group(self.player)
        
    def update(self, surface, keys, current_time,show_game=True):
        self.last_game_data = None;
        self.current_time = current_time

        self.handle_states(keys)
        
        if (c.GRAPHICS_SETTINGS != c.NONE and show_game):
            self.draw(surface)
     
    
    def handle_states(self, keys):
        
        self.update_all_sprites(keys)

        if (pg.K_PAGEUP in keys and keys[pg.K_PAGEUP] is not None and keys[pg.K_PAGEUP]):
            if (not self.last_save):
                self.save_internal_state();
                print('state saved');
            self.last_save = True;
        else:
            self.last_save = False;
        if (pg.K_PAGEDOWN in keys and keys[pg.K_PAGEDOWN] is not None and keys[pg.K_PAGEDOWN]):
            if (not self.last_load):
                self.load_internal_state();
                print('state loaded');
            self.last_load = True;
        else:
            self.last_load = False;            

        

    
    def update_all_sprites(self, keys):
        time_info = {c.CURRENT_TIME:self.current_time};
        
        if self.player.dead:
            self.player.update(keys, time_info, self.powerup_group)
            if self.current_time - self.death_timer >= 0:
                self.update_game_info()
                self.done = True
        elif self.player.state == c.IN_CASTLE:
            self.player.update(keys, time_info, None)
            self.flagpole_group.update()
            if self.current_time - self.castle_timer > 2000:
                self.update_game_info()
                self.done = True
        elif self.in_frozen_state():
            self.player.update(keys, time_info, None)
            self.check_checkpoints()
            self.update_viewport()

        else:

            self.player.update(keys, time_info, self.powerup_group)
            self.flagpole_group.update()
            self.check_checkpoints()
            self.slider_group.update()
            self.enemy_group.update(time_info, self)
            self.shell_group.update(time_info, self)
            self.brick_group.update()
            self.box_group.update(time_info)
            self.powerup_group.update(time_info, self)
            self.coin_group.update(time_info)
            self.brickpiece_group.update()
            self.dying_group.update(time_info, self)

            self.update_player_position() #2 dicts, 1 group within
            self.check_for_player_death()
            self.check_task()
            self.update_viewport()


    
    def check_checkpoints(self):
        checkpoint = pg.sprite.spritecollideany(self.player, self.checkpoint_group)
        
        if checkpoint:
            if checkpoint.type == c.CHECKPOINT_TYPE_ENEMY:
                group = self.enemy_group_list[checkpoint.enemy_groupid]
                self.enemy_group.add(group)
            elif checkpoint.type == c.CHECKPOINT_TYPE_FLAG:
                self.player.state = c.FLAGPOLE
                if self.player.rect.bottom < self.flag.rect.y:
                    self.player.rect.bottom = self.flag.rect.y
                    self.player.update_hitbox();
                self.flag.state = c.SLIDE_DOWN
                self.update_flag_score()
            elif checkpoint.type == c.CHECKPOINT_TYPE_CASTLE:
                self.player.state = c.IN_CASTLE
                self.player.x_vel = 0
                self.castle_timer = self.current_time
                self.flagpole_group.add(stuff.CastleFlag(8745, 322))
            elif (checkpoint.type == c.CHECKPOINT_TYPE_MUSHROOM and
                    self.player.y_vel < 0):
                mushroom_box = box.Box(checkpoint.rect.x, checkpoint.rect.bottom - 40,
                                c.TYPE_LIFEMUSHROOM, self.powerup_group)
                mushroom_box.start_bump([])
                self.box_group.add(mushroom_box)
                self.player.y_vel = 7
                self.player.rect.y = mushroom_box.rect.bottom
                self.player.update_hitbox();
                self.player.state = c.FALL
            elif checkpoint.type == c.CHECKPOINT_TYPE_PIPE:
                self.player.state = c.WALK_AUTO
            elif checkpoint.type == c.CHECKPOINT_TYPE_PIPE_UP:
                self.change_map(checkpoint.map_index, checkpoint.type)
            elif checkpoint.type == c.CHECKPOINT_TYPE_MAP:
                self.change_map(checkpoint.map_index, checkpoint.type)
            elif checkpoint.type == c.CHECKPOINT_TYPE_BOSS:
                self.player.state = c.WALK_AUTO
            checkpoint.kill()

    def update_flag_score(self):
        base_y = c.GROUND_HEIGHT - 80
        
        y_score_list = [(base_y, 100), (base_y-120, 400),
                    (base_y-200, 800), (base_y-320, 2000),
                    (0, 5000)]
        for y, score in y_score_list:
            if self.player.rect.y > y:
                #self.update_score(score, self.flag)
                break
        
    def update_player_position(self):
        if self.player.state == c.UP_OUT_PIPE:
            return

        
        self.player.hitbox.x += round(self.player.x_vel)
        if self.player.hitbox.left < self.map_bounds[0]:
            self.player.hitbox.left = self.map_bounds[0];
        elif self.player.hitbox.right > self.map_bounds[1]:
            self.player.hitbox.right = self.map_bounds[1];
        self.check_player_x_collisions()

        

        if not self.player.dead:

            self.player.hitbox.y += round(self.player.y_vel)
            self.check_player_y_collisions()
            if self.player.hitbox.top < self.map_bounds[2]:
                self.player.hitbox.top = self.map_bounds[2];
        


        self.player.rect_from_hitbox();

        
        
    


    def check_player_x_collisions(self):
        ground_step_pipe = pg.sprite.spritecollideany(self.player, self.ground_step_pipe_group,hitbox_collide)
        brick = pg.sprite.spritecollideany(self.player, self.brick_group,hitbox_collide)
        box = pg.sprite.spritecollideany(self.player, self.box_group,hitbox_collide)
        enemy = pg.sprite.spritecollideany(self.player, self.enemy_group,hitbox_collide)
        shell = pg.sprite.spritecollideany(self.player, self.shell_group,hitbox_collide)
        powerup = pg.sprite.spritecollideany(self.player, self.powerup_group,hitbox_collide)

        if box:
            self.adjust_player_for_x_collisions(box)
        elif brick:
            self.adjust_player_for_x_collisions(brick)
        elif ground_step_pipe:
            if (ground_step_pipe.name == c.MAP_PIPE and
                ground_step_pipe.type == c.PIPE_TYPE_HORIZONTAL):
                return
            self.adjust_player_for_x_collisions(ground_step_pipe)
        elif powerup:
            if powerup.type == c.TYPE_MUSHROOM:
                #self.update_score(1000, powerup, 0)
                if not self.player.big:
                    self.player.y_vel = -1
                    self.player.state = c.SMALL_TO_BIG
            elif powerup.type == c.TYPE_FIREFLOWER:
                #self.update_score(1000, powerup, 0)
                if not self.player.big:
                    self.player.state = c.SMALL_TO_BIG
                elif self.player.big and not self.player.fire:
                    self.player.state = c.BIG_TO_FIRE
            elif powerup.type == c.TYPE_STAR:
                #self.update_score(1000, powerup, 0)
                self.player.invincible = True
            #elif powerup.type == c.TYPE_LIFEMUSHROOM:
                #self.update_score(500, powerup, 0)
                #self.game_info[c.LIVES] += 1
            if powerup.type != c.TYPE_FIREBALL:
                powerup.kill()
        elif enemy:
            if self.player.invincible:
                #self.update_score(100, enemy, 0)
                self.move_to_dying_group(self.enemy_group, enemy)
                direction = c.RIGHT if self.player.facing_right else c.LEFT
                enemy.start_death_jump(direction)
            elif self.player.hurt_invincible:
                pass
            elif self.player.big:
                self.player.y_vel = -1
                self.player.state = c.BIG_TO_SMALL
            else:
                self.player.die()
        elif shell:
            if shell.state == c.SHELL_SLIDE:
                if self.player.invincible:
                    #self.update_score(200, shell, 0)
                    self.move_to_dying_group(self.shell_group, shell)
                    direction = c.RIGHT if self.player.facing_right else c.LEFT
                    shell.start_death_jump(direction)
                elif self.player.hurt_invincible:
                    pass
                elif self.player.big:
                    self.player.y_vel = -1
                    self.player.state = c.BIG_TO_SMALL
                else:
                    self.player.die()
            else:
                #self.update_score(400, shell, 0)
                if self.player.hitbox.x < shell.rect.x:
                    self.player.hitbox.left = shell.rect.x 
                    shell.direction = c.RIGHT
                    shell.x_vel = 10
                else:
                    self.player.hitbox.x = shell.rect.left
                    shell.direction = c.LEFT
                    shell.x_vel = -10
                shell.rect.x += shell.x_vel * 4
                shell.state = c.SHELL_SLIDE


    def adjust_player_for_x_collisions(self, collider):
        if collider.name == c.MAP_SLIDER:
            return

        if self.player.hitbox.x < collider.rect.x:
            self.player.hitbox.right = collider.rect.left
        else:
            self.player.hitbox.left = collider.rect.right
        self.player.x_vel = 0

    def check_player_y_collisions(self):
        ground_step_pipe = pg.sprite.spritecollideany(self.player, self.ground_step_pipe_group,hitbox_collide)
        enemy = pg.sprite.spritecollideany(self.player, self.enemy_group,hitbox_collide)
        shell = pg.sprite.spritecollideany(self.player, self.shell_group,hitbox_collide)

        # decrease runtime delay: when player is on the ground, don't check brick and box
        if self.player.rect.bottom < c.GROUND_HEIGHT:
            brick = pg.sprite.spritecollideany(self.player, self.brick_group,hitbox_collide)
            box = pg.sprite.spritecollideany(self.player, self.box_group,hitbox_collide)
            brick, box = self.prevent_collision_conflict(brick, box)
        else:
            brick, box = False, False

        if box:
            self.adjust_player_for_y_collisions(box)
        elif brick:
            self.adjust_player_for_y_collisions(brick)
        elif ground_step_pipe:
            self.adjust_player_for_y_collisions(ground_step_pipe)
        elif enemy:
            if self.player.invincible:
                #self.update_score(100, enemy, 0)
                self.move_to_dying_group(self.enemy_group, enemy)
                direction = c.RIGHT if self.player.facing_right else c.LEFT
                enemy.start_death_jump(direction)
            elif (enemy.name == c.PIRANHA or
                enemy.name == c.FIRESTICK or
                enemy.name == c.FIRE_KOOPA or
                enemy.name == c.FIRE):
                pass
            elif self.player.y_vel > 0:
                #self.update_score(100, enemy, 0)
                enemy.state = c.JUMPED_ON
                if enemy.name == c.GOOMBA:
                    self.move_to_dying_group(self.enemy_group, enemy)
                elif enemy.name == c.KOOPA or enemy.name == c.FLY_KOOPA:
                    self.enemy_group.remove(enemy)
                    self.shell_group.add(enemy)

                self.player.hitbox.bottom = enemy.rect.top
                self.player.state = c.JUMP
                self.player.y_vel = -7
        elif shell:
            #TODO: check if +7/-7 should be scaled based on player size
            if self.player.y_vel > 0:
                if shell.state != c.SHELL_SLIDE:
                    shell.state = c.SHELL_SLIDE
                    if self.player.hitbox.centerx < shell.rect.centerx:
                        shell.direction = c.RIGHT
                        shell.rect.left = self.player.hitbox.right + 7
                    else:
                        shell.direction = c.LEFT
                        shell.rect.right = self.player.hitbox.left - 7

        self.check_is_falling(self.player)
        
        self.check_if_player_on_IN_pipe()
    
    def prevent_collision_conflict(self, sprite1, sprite2):
        if sprite1 and sprite2:
            distance1 = abs(self.player.rect.centerx - sprite1.rect.centerx)
            distance2 = abs(self.player.rect.centerx - sprite2.rect.centerx)
            if distance1 < distance2:
                sprite2 = False
            else:
                sprite1 = False
        return sprite1, sprite2
        
    def adjust_player_for_y_collisions(self, sprite):
        if self.player.rect.top > sprite.rect.top:
            if sprite.name == c.MAP_BRICK:
                self.check_if_enemy_on_brick_box(sprite)
                if sprite.state == c.RESTING:
                    if self.player.big and sprite.type == c.TYPE_NONE:
                        sprite.change_to_piece(self.dying_group)
                    else:
                        #if sprite.type == c.TYPE_COIN:
                            #self.update_score(200, sprite, 1)
                        sprite.start_bump([])
            elif sprite.name == c.MAP_BOX:
                self.check_if_enemy_on_brick_box(sprite)
                if sprite.state == c.RESTING:
                    #if sprite.type == c.TYPE_COIN:
                        #self.update_score(200, sprite, 1)
                    sprite.start_bump([])
            elif (sprite.name == c.MAP_PIPE and
                sprite.type == c.PIPE_TYPE_HORIZONTAL):
                return
            
            self.player.y_vel = 7
            self.player.hitbox.top = sprite.rect.bottom
            self.player.state = c.FALL
        else:
            self.player.y_vel = 0
            self.player.hitbox.bottom = sprite.rect.top
            if self.player.state == c.FLAGPOLE:
                self.player.state = c.WALK_AUTO
            elif self.player.state == c.END_OF_LEVEL_FALL:
                self.player.state = c.WALK_AUTO
            else:
                self.player.state = c.WALK
    
    def check_if_enemy_on_brick_box(self, brick):
        brick.rect.y -= 5
        enemy = pg.sprite.spritecollideany(brick, self.enemy_group)
        if enemy:
            #self.update_score(100, enemy, 0)
            self.move_to_dying_group(self.enemy_group, enemy)
            if self.player.rect.centerx > brick.rect.centerx:
                direction = c.RIGHT
            else:
                direction = c.LEFT
            enemy.start_death_jump(direction)
        brick.rect.y += 5

    def in_frozen_state(self):
        if (self.player.state == c.SMALL_TO_BIG or
            self.player.state == c.BIG_TO_SMALL or
            self.player.state == c.BIG_TO_FIRE or
            self.player.state == c.DEATH_JUMP or
            self.player.state == c.DOWN_TO_PIPE or
            self.player.state == c.UP_OUT_PIPE):
            return True
        else:
            return False

    def check_is_falling(self, sprite):
        sprite.rect.y += 1

        if (pg.sprite.spritecollideany(sprite, self.ground_step_pipe_group) is None and
            pg.sprite.spritecollideany(sprite,self.brick_group) is None and
            pg.sprite.spritecollideany(sprite,self.box_group) is None):
            if (sprite.state == c.WALK_AUTO or
                sprite.state == c.END_OF_LEVEL_FALL):
                sprite.state = c.END_OF_LEVEL_FALL
            elif (sprite.state != c.JUMP and 
                sprite.state != c.FLAGPOLE and
                not self.in_frozen_state()):
                sprite.state = c.FALL
        sprite.rect.y -= 1

    
    def check_for_player_death(self):
        if (self.player.rect.y > self.map_bounds[3]):
            self.player.die();
        elif self.task_bounds is not None:
            if not ((self.player.rect.centerx < self.task_bounds[1] and self.player.rect.centerx>self.task_bounds[0]) and (self.player.rect.centery < self.task_bounds[3] and self.player.rect.centery > self.task_bounds[2])):
                self.player.die();

    def check_task(self):
        if self.task_path is None:
            if self.task is not None and self.player.hitbox.collidepoint(self.task):
                self.done = True;
                self.task_reached = 1;
        elif self.task is not None:
            p = self.player.rect.center;
            if len(self.task_path) > 1:
                if math.sqrt((p[0]-self.task[0])**2 + (p[1]-self.task[1])**2) < TASK_PATH_CHANGE_THRESHOLD:
                    self.increment_task_path();
            elif self.player.hitbox.collidepoint(self.task):
                self.increment_task_path(load_next=False);
                
                self.done = True;

    def increment_task_path(self,load_next=True,move_player=False):
        assert self.task_path is not None
        assert self.task is not None

        if (len(self.task_path) == 0):
            raise Exception("Smbpy Segment Error: tried to increment task path with nothing left on the path");
        if move_player:
            self.player.rect.centerx = self.task[0];
            self.player.rect.centery = self.task[1];
        self.task_path.remove(self.task);
        self.task_reached += 1;
        if load_next:
            self.task = self.task_path[0];



    def check_if_player_on_IN_pipe(self):
        '''check if player is on the pipe which can go down in to it '''
        self.player.rect.y += 1
        pipe = pg.sprite.spritecollideany(self.player, self.pipe_group)
        if pipe and pipe.type == c.PIPE_TYPE_IN:
            if (self.player.crouching and
                self.player.rect.x < pipe.rect.centerx and
                self.player.rect.right > pipe.rect.centerx):
                self.player.state = c.DOWN_TO_PIPE
        self.player.rect.y -= 1
        
    def update_game_info(self):
        #if self.player.dead:
#            self.persist[c.LIVES] -= 1

        #if self.persist[c.LIVES] == 0:
            #self.next = c.GAME_OVER
        #elif self.overhead_info.time == 0:
            #self.next = c.TIME_OUT
        #elif self.player.dead:
            #self.next = c.LOAD_SCREEN
        #else:
            #self.game_info[c.LEVEL_NUM] += 1
            #self.next = c.LOAD_SCREEN
        self.next = self;

    def update_viewport(self):
        x_third = self.viewport.x + self.viewport.w//3
        y_third = self.viewport.bottom - self.viewport.h//3;
        player_centerx = self.player.rect.centerx;
        player_centery = self.player.rect.centery
        
        if (self.player.x_vel > 0 and 
            player_centerx >= x_third and
            self.viewport.right < self.map_bounds[1]):
            self.viewport.x += round(self.player.x_vel);
        elif self.player.x_vel < 0 and self.viewport.x > self.map_bounds[0]:
            self.viewport.x += round(self.player.x_vel);

        if (self.player.y_vel < 0 and
            player_centery <= y_third and
            self.viewport.top > self.map_bounds[2]):
            self.viewport.y += round(self.player.y_vel);
        elif self.player.y_vel > 0 and self.viewport.bottom < self.map_bounds[3]:
            self.viewport.y += round(self.player.y_vel);


    
    def move_to_dying_group(self, group, sprite):
        group.remove(sprite)
        self.dying_group.add(sprite)
        
    def update_score(self, score, sprite, coin_num=0):
        #self.game_info[c.SCORE] += score
        #self.game_info[c.COIN_TOTAL] += coin_num
        #x = sprite.rect.x
        #y = sprite.rect.y - 10
#        self.moving_score_list.append(stuff.Score(x, y, score))
        print('score updated')

    def clip_rect(self,rect:pg.rect.Rect): #clip to positive this is so stupid
        x = max(rect.x,0);
        y = max(rect.y,0);
        w = rect.width - x + rect.x;
        h = rect.height - y + rect.y;
        rect.x = x;
        rect.y = y;
        rect.w = w;
        rect.h = h;

    def draw(self, surface):
        if not self.bg_image and self.background is not None:
            self.ground_step_pipe_group.draw(self.background);
            self.bg_image = True;
            # print("drawing background");

        if (self.background is not None):
            self.level.blit(self.background, self.viewport, self.viewport)

        self.powerup_group.draw(self.level)
        self.brick_group.draw(self.level)
        self.box_group.draw(self.level)
        if (c.GRAPHICS_SETTINGS >= c.MED):
            self.coin_group.draw(self.level)
            self.dying_group.draw(self.level)
            self.brickpiece_group.draw(self.level)
            self.flagpole_group.draw(self.level)
        self.shell_group.draw(self.level)
        self.enemy_group.draw(self.level)
        self.player_group.draw(self.level)
        self.slider_group.draw(self.level)

        if c.DEBUG:
            self.ground_step_pipe_group.draw(self.level)
            self.checkpoint_group.draw(self.level)

        if self.task is not None:
            shader_surface = pg.surface.Surface(self.level.get_size(),pg.SRCALPHA);
            if self.task_bounds is not None:
                shader_surface.fill((0,0,0,c.SHADER_ALPHA));
                bounds_rect = pg.rect.Rect(self.task_bounds[0],self.task_bounds[2],self.task_bounds[1]-self.task_bounds[0],self.task_bounds[3]-self.task_bounds[2]);
                shader_surface.fill((0,0,0,0),bounds_rect);
            if self.task_path is not None and len(self.task_path) > 1:
                pg.draw.circle(shader_surface,(0,127,255,c.SHADER_ALPHA),self.task_path[1],c.TILE_SIZE/4);
            pg.draw.circle(shader_surface,(0,255,0,c.SHADER_ALPHA),self.task,c.TILE_SIZE/4);
            
            # task_rect = pg.rect.Rect(self.task[0]-c.TILE_SIZE/2,self.task[1]-c.TILE_SIZE/2,c.TILE_SIZE,c.TILE_SIZE);
            # self.clip_rect(task_rect);
            # shader_surface.fill((0,255,0,c.SHADER_ALPHA),task_rect);

            self.level.blit(shader_surface,self.viewport.topleft,self.viewport);
            

        
        pg.draw.circle(self.level,c.PURPLE,self.map_list[0][1],2/16*c.TILE_SIZE);

        # surface.fill(c.BLACK);
        surface.blit(self.level, (0,0), (self.viewport.x,self.viewport.y,self.viewport.width,self.viewport.height+40));

        # print(self.last_game_data);
        if c.DRAW_GRID and self.last_grid:
            # print('drawing grid');
            grid = self.last_grid;
            grid_surface = pg.surface.Surface((len(grid)*2,len(grid[0])*2),pg.SRCALPHA);
            grid_surface.fill((0,0,0,c.SHADER_ALPHA));
            for i,row in enumerate(grid):
                for j,el in enumerate(row):
                    if el:
                        pg.draw.rect(grid_surface,(255,255,255,c.SHADER_ALPHA),(i*2,j*2,2,2));
            surface.blit(grid_surface,(20,20));

    def get_map_data(self,tile_scale): #SHOULD NOT BE CALLED DURING PLAY AS WILL CREATE SIGNIFICANT LAG CONFLICT WITH GET_GAME_DATA
        bounds = self.map_bounds;
        center = (bounds[0]/2 + bounds[1]/2, bounds[2]/2 + bounds[3]/2);
        size =  (bounds[1]-bounds[0],bounds[3]-bounds[2]);
        self.update_rect_grid(tile_scale,center,size);

        return JITDict(delegates={
            'grid_bounds':lambda:bounds,
            'enemy_grid':self.get_enemy_grid,
            'collision_grid': self.get_collision_grid,
            'powerup_grid':self.get_powerup_grid,
            'box_grid':self.get_box_grid,
            'brick_grid':self.get_brick_grid,
        })


    #view distance: number of tiles in each direction. EX: view distance of 3.5 would form a 7x7 tile grid with the requisite number of subdivisions as dictated by tile_scale
    def get_game_data(self,view_distance,tile_scale,obstruction=False):

        if (self.last_game_data is not None):
            return self.last_game_data;

        self.no_obstruction = not obstruction;
        self.update_rect_grid(tile_scale,self.player.rect.center,(view_distance*2*c.TILE_SIZE,view_distance*2*c.TILE_SIZE));
        
        self.last_game_data = JITDict[str,Any](delegates=
        {
            'task_reached': lambda:self.task_reached,
            'task_path_remaining': lambda:None if self.task_path is None else len(self.task_path),
            'task_path_complete': lambda:self.task_path is not None and len(self.task_path) == 0,
            'done': lambda:self.done,
            'task_position_offset': lambda:[self.task[0]-self.player.rect.centerx,self.task[1]-self.player.rect.centery] if self.task is not None else None,
            'task_position': lambda:self.task,
            'pos': lambda:[self.player.rect.centerx,self.player.rect.centery],
            'vel': lambda:[self.player.x_vel,self.player.y_vel],
            'player_state': self.player.get_powerup_state,
            'enemy_grid': self.get_enemy_grid,
            'collision_grid': self.get_collision_grid,
            'powerup_grid': self.get_powerup_grid,
            'box_grid': self.get_box_grid,
            'brick_grid': self.get_brick_grid,
            'task_obstructions': self.get_task_obstructions});

        return self.last_game_data;


    #view distance: integer number of subdivided tiles in each direction, excluding center. EX: 3 would be a 7x7
    #tile scale: power of two: number of subdivisions per tile length
    def update_rect_grid(self,tile_scale,center,size):
        if self.grid_rects is None or size != self.grid_size:
            rect_width = c.TILE_SIZE/tile_scale;
            self.grid_center = center;
            self.grid_size = size;
            x_rects = int(size[0]/rect_width);
            y_rects = int(size[1]/rect_width);
            x_lefts = range(int(-x_rects/2),int(x_rects/2)) if x_rects % 2 == 0 else [x - 0.5 for x in range(math.ceil(-x_rects/2),math.ceil(x_rects/2))];
            y_tops = range(int(-y_rects/2),int(y_rects/2)) if y_rects % 2 == 0 else [y - 0.5 for y in range(math.ceil(-y_rects/2),math.ceil(y_rects/2))];

            self.grid_rects = [[pg.Rect(center[0] + i * rect_width, center[1] + j * rect_width,rect_width,rect_width) for j in y_tops] for i in x_lefts];
        else:
            offset = [center[0] - self.grid_center[0],center[1] - self.grid_center[1]];
            self.grid_center = center;
            [[rect.move_ip(offset[0],offset[1]) for rect in row] for row in self.grid_rects];


    def get_enemy_grid(self)->list[list[int]]:
        spriteRects = [sprite.rect for sprite in self.enemy_group];
        return [[1 if rect.collidelist(spriteRects) >= 0 else 0 for rect in row] for row in self.grid_rects];

    def get_collision_grid(self)->list[list[int]]:
        spriteRects = [[sprite.rect for sprite in group] for group in [self.ground_step_pipe_group, self.brick_group, self.box_group]];
        res = [[1 if any(rect.collidelist(sprites) >= 0 for sprites in spriteRects) else 0 for rect in row] for row in self.grid_rects];
        if c.DRAW_GRID:
            self.last_grid = res;
        return res;

    def get_powerup_grid(self)->list[list[int]]:
        spriteRects = [sprite.rect for sprite in self.powerup_group];
        return [[1 if rect.collidelist(spriteRects) >= 0 else 0 for rect in row] for row in self.grid_rects];

    def get_box_grid(self)->list[list[int]]:
        spriteRects = [sprite.rect for sprite in self.box_group];
        return [[1 if rect.collidelist(spriteRects) >= 0 else 0 for rect in row] for row in self.grid_rects];

    def get_brick_grid(self)->list[list[int]]:
        spriteRects = [sprite.rect for sprite in self.brick_group];
        return [[1 if rect.collidelist(spriteRects) >= 0 else 0 for rect in row] for row in self.grid_rects];


    #return [distance,blocks,enemies], with [blocks,enemies] counting the number of objects between the player's center and the task. For fitness purposes only

    #TODO: Examine performance impact of all of this *math*
    def get_task_obstructions(self)->list[float]:
        center = self.player.rect.center;
        distance = math.sqrt((center[0]-self.task[0])**2 + (center[1]-self.task[1])**2);
        if self.no_obstruction:
            return [distance,0,0];
        block_obstructions = 0;
        check_group = pg.sprite.Group(self.ground_step_pipe_group,
                    self.brick_group, self.box_group);
        for sprite in check_group:
            block_obstructions += 1 if collideRectLine(sprite.rect,center,self.task) else 0;

        enemy_obstructions = 0;
        for sprite in self.enemy_group:
            enemy_obstructions += 1 if collideRectLine(sprite.rect,center,self.task) else 0;
        return [distance,block_obstructions,enemy_obstructions];


def collideLineLine(l1_p1, l1_p2, l2_p1, l2_p2):
    
    # normalized direction of the lines and start of the lines
    P  = pg.math.Vector2(*l1_p1)
    line1_vec = pg.math.Vector2(*l1_p2) - P
    R = line1_vec.normalize()
    Q  = pg.math.Vector2(*l2_p1)
    line2_vec = pg.math.Vector2(*l2_p2) - Q
    S = line2_vec.normalize()

    # normal vectors to the lines
    RNV = pg.math.Vector2(R[1], -R[0])
    SNV = pg.math.Vector2(S[1], -S[0])
    RdotSVN = R.dot(SNV)
    if RdotSVN == 0:
        return False

    # distance to the intersection point
    QP  = Q - P
    t = QP.dot(SNV) / RdotSVN 
    u = QP.dot(RNV) / RdotSVN 

    return t > 0 and u > 0 and t*t < line1_vec.magnitude_squared() and u*u < line2_vec.magnitude_squared()

def collideRectLine(rect, p1, p2):

    return (collideLineLine(p1, p2, rect.topleft, rect.bottomleft) or
            collideLineLine(p1, p2, rect.bottomleft, rect.bottomright) or
            collideLineLine(p1, p2, rect.bottomright, rect.topright) or
            collideLineLine(p1, p2, rect.topright, rect.topleft))

        
def hitbox_collide(sprite,other):
    return sprite.hitbox.colliderect(other.rect);