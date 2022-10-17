import os
from threading import Event
from pathlib import Path
import time
from typing import Iterable
from baseGame import RunGame
from gameReporting import ThreadedGameReporter
from game_runner_neat import Genome_Executor
from games.smb1Py.py_mario_bros.PythonSuperMario_master.smb_game import SMB1Game

from games.smb1Py.py_mario_bros.PythonSuperMario_master.source import constants as c, setup, tools
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import Segment, SegmentState

class LevelRenderer:
    def __init__(self,state:SegmentState,
            max_dims:tuple[int,int]=(1000,500),
            reached_color=c.GREEN,
            failed_color=c.RED,
            path_color=c.BLUE,
            active_path_color=c.YELLOW,
            completed_path_color=c.GREEN,
            failed_path_color=c.RED,
            point_size=4, #all sizes POST downscaling
            path_width=2,
            active_path_width=None,
            completed_path_width=None,
            failed_path_width=None,
            start_render=False):
        
        self.state = state;
        self.game = None;

        self.dims = max_dims
        self.display_set = False;
        
        if start_render:
            self.setup_display();
        
        self.last_level_dims = None;
        self.last_scale = None;
        
        self.reached:Iterable[tuple[float,float]] = [];
        self.failed:Iterable[tuple[float,float]] = [];
        self.paths:dict[int,Iterable[tuple[float,float]]];
        self.active_paths:Iterable[int] = [];
        self.completed_paths:Iterable[int] = [];
        self.failed_paths:Iterable[int] = [];

        self.reached_color = reached_color;
        self.failed_color = failed_color;
        self.path_color = path_color;
        self.active_color = active_path_color;
        self.failed_path_color = failed_path_color;
        self.completed_path_color = completed_path_color;

        self.point_size=point_size;
        self.path_width=path_width;
        self.active_width= path_width if active_path_width is None else active_path_width;
        self.completed_width = path_width if completed_path_width is None else completed_path_width;
        self.failed_width = path_width if failed_path_width is None else failed_path_width;


    def setup_display(self):
        import pygame as pg
        print("display startup...");

        setup.get_GFX();
        self.game = Segment()
        self.game.startup(0,{c.LEVEL_NUM:1},initial_state=self.state);

        pg.display.set_mode(self.dims);
        pg.display.set_caption("Level Preview");
        self.screen = pg.display.get_surface();

        self.display_set = True;

    def set_annotations(self,
            reached:Iterable[tuple[float,float]],
            failed:Iterable[tuple[float,float]],
            paths:dict[int,Iterable[tuple[float,float]]],
            active_paths:Iterable[int]=[],
            failed_paths:Iterable[int]=[],
            completed_paths:Iterable[int]=[],
                ):
        self.reached = reached;
        self.failed = failed;
        self.paths = paths;
        self.active_paths = active_paths;
        self.failed_paths = failed_paths;
        self.completed_paths = completed_paths;

    def set_paths(self,paths:dict[int,Iterable[tuple[float,float]]]):
        self.paths = paths;

    def update_active_paths(self,active_paths:Iterable[int]):
        self.active_paths = active_paths;

    def update_failed_paths(self,failed_paths:Iterable[int]):
        self.failed_paths=failed_paths;

    def update_completed_paths(self,completed_paths:Iterable[int]):
        self.completed_paths=completed_paths;

    def display(self):
        import pygame as pg
        if not self.display_set:
            self.setup_display();
        assert self.game is not None

        level = pg.Surface(self.game.level.get_size());
        self.game.draw(level,blit_all=True);

        if self.last_level_dims != level.get_size() or self.last_scale is None:
            self.last_level_dims = level.get_size();
            self.last_scale = min([self.dims[i]/self.last_level_dims[i] for i in [0,1]]);
        scale = self.last_scale;
        out_dim = [int(scale*self.last_level_dims[i]) for i in [0,1]];


        for p in self.reached:
            pg.draw.circle(level,self.reached_color,p,self.point_size/scale);

        for p in self.failed:
            pg.draw.circle(level,self.failed_color,p,self.point_size/scale);


        special_paths = set(self.active_paths) | set(self.completed_paths) | set(self.failed_paths);
        for id,path in self.paths.items():
            if id in special_paths:
                continue;
            color = self.path_color;
            width = self.path_width;

            pg.draw.lines(level,color,False,path,width=width);

        for id in self.failed_paths:
            color = self.failed_path_color;
            width = self.failed_width;

            pg.draw.lines(level,color,False,self.paths[id],width=width);

        for id in self.completed_paths:
            color = self.completed_path_color;
            width = self.completed_width;

            pg.draw.lines(level,color,False,self.paths[id],width=width);

        for id in self.active_paths:
            color = self.active_color;
            width = self.active_width;

            pg.draw.lines(level,color,False,self.paths[id],width=width);


        self.screen.blit(pg.transform.smoothscale(level,out_dim),(0,0));

        pg.display.flip();

class PathMessage:
    ACTIVE_CHANGED=0;
    PATH_COMPLETED=1;
    def __init__(self,pid:int,message_type:int,path_id:int):
        self.pid = pid;
        self.type = message_type;
        self.path_id = path_id;

    @classmethod
    def complete(cls,pid,path_id):
        return PathMessage(pid,cls.PATH_COMPLETED,path_id);

    @classmethod
    def active(cls,pid,path_id):
        return PathMessage(pid,cls.ACTIVE_CHANGED,path_id);

class LevelRendererReporter(ThreadedGameReporter[PathMessage]): #process_num,active_id
    
    def __init__(self):
        super().__init__();
        self.active_id = None;
        self.id_complete = False;
        self.reset_paths();

    def reset_paths(self):
        self.active:dict[int,int] = {};
        self.completed:set[int] = set();
        self.failed:set[int] = set();
    
    def on_training_data_load(self, game: SMB1Game, id):
        self.pid = os.getpid();
        if (self.active_id != id): #active changed
            self.put_data(PathMessage.active(self.pid,id));
        self.active_id  = id;
        self.id_complete = False;
        
    def on_finish(self, game: RunGame):
        if not self.id_complete and game.getMappedData()['task_path_complete']:
            self.put_data(PathMessage.complete(self.pid,self.active_id));
            self.id_complete = True;

    def update_display(self,renderer:LevelRenderer):
        #pull all updates from the pool
        for message in self.get_all_data():
            pid = message.pid;
            did = message.path_id; #data id
            if did not in renderer.paths: #renderer no longer rendering level
                prev = self.active[pid];
                if prev not in self.completed:
                    self.failed.add(prev);
                    renderer.update_failed_paths(self.failed);
                self.active.pop(pid);
                renderer.update_active_paths(self.active.values());
                continue;
            match message.type:
                case PathMessage.ACTIVE_CHANGED:
                    if pid not in self.active:
                        self.active[pid] = did;
                        renderer.update_active_paths(self.active.values());
                        break;

                    prev = self.active[pid];
                    if prev == did: 
                        print("LevelRendererReporter warning: Unnecessary data transfer - changing active id to already active path for process",pid);
                        break;
                    if prev not in self.completed:
                        self.failed.add(prev);
                        renderer.update_failed_paths(self.failed);
                    self.active[pid] = did;
                    renderer.update_active_paths(self.active.values());
                case PathMessage.PATH_COMPLETED:
                    if pid not in self.active:
                        self.active[pid] = did;
                        renderer.update_active_paths(self.active.values());
                    if did != self.active[pid]:
                        print("LevelRendererReporter error: completed path does not match active path for process",pid);
                        # break;
                    if did in self.completed:
                        print("LevelRendererReporter warning: Unnecessary data transfer - completing previously completed path")
                    self.completed.add(did);
                    renderer.update_completed_paths(self.completed);
        renderer.display();

    def render_loop(self,renderer:LevelRenderer,kill_event:Event,interval=5): #update interval in seconds
        import pygame as pg
        while not kill_event.is_set():
            self.update_display(renderer);
            # print(renderer.display_set);
            pg.event.pump();
            time.sleep(interval);




if __name__ == "__main__":
    import pygame as pg
    level_path = Path('levels')/'testing'/'test1.lvl';
    level = SegmentState(None,None,file_path=level_path);


    display = LevelRenderer(level);

    display.set_annotations([(10,10),(30,20)],[],{0:((50,50),(60,70),(100,45))});

    display.display();

    while True:
        evs = pg.event.get();
        if (pg.WINDOWCLOSE in [ev.type for ev in evs]):
            exit();