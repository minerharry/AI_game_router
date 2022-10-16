from __future__ import annotations
from abc import ABC, abstractmethod
import abc
import random
from typing import Iterable, Type
from PIL import Image, ImageDraw, ImageFont
import math
import collections.abc
import numpy as np
from gameReporting import GameReporter
# from gameReporting import GameReporter

from runnerConfiguration import RunnerConfig

def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def listifyArray(array):
    return list(flatten(array));


class EvalGame:
    def __init__(self,gameClass:Type[RunGame],**kwargs):
        self.gameClass = gameClass;
        self.initInputs = kwargs;
        self.reporters:list[GameReporter] = [];

    def register_reporter(self,reporter:GameReporter):
        if reporter not in self.reporters:
            print(f"EvalGame: reporter {reporter} registered")
            self.reporters.append(reporter);

    def deregister_reporter(self,reporter):
        if reporter in self.reporters:
            self.reporters.remove(reporter);    

    def start(self,runnerConfig:RunnerConfig,**kwargs):
        if kwargs is not None:
            for name,arg in kwargs.items():
                self.initInputs[name] = arg;
        game = self.gameClass(runnerConfig,reporters=self.reporters,**self.initInputs);
        return game;
        

class RunGame(ABC):
    def __init__(self,runnerConfig:RunnerConfig,reporters:list|None=None,**kwargs):
        self.steps = 0;
        self.runConfig = runnerConfig;
        self.reporters:list[GameReporter] = [];
        self.mapDataCache = None;
        if reporters:
            [self.register_reporter(rep) for rep in reporters];
            self.signal_start();
        if ('training_datum_id' in kwargs):
            self.signal_training_data_load(kwargs['training_datum_id'])

    def getData(self)->list:
        mappedData = self.getMappedData();
        returnData = self.runConfig.returnData;
        result = [];
        for datum in returnData:
            if (isinstance(datum,str)):
                result.append(mappedData.get(datum));
            else:
                result += datum.extract_data(mappedData.get(datum.name));
            # elif (datum.data_type == 'array'):
            #     if (datum.name not in mappedData):
            #         print(datum.name);
            #     result += listifyArray(mappedData.get(datum.name));
            # elif (datum.data_type == 'ndarray'):
            #     result += mappedData.get(datum.name).tolist();
        
        return result;

    def register_reporter(self,reporter:GameReporter):
        self.reporters.append(reporter);

    def getFitnessScore(self):
        return self.runConfig.fitnessFromGameData(self.getMappedData());

    def getMappedData(self)->dict:
        mappedData = self.getOutputData();
        mappedData['steps'] = self.steps;
        self.mapDataCache = mappedData;
        return mappedData;

    @abstractmethod
    def getOutputData(self)->dict:
        #return dict of all data available from game, sans 'steps'
        pass;

    def tickInput(self,inputs):
        self.processInput(inputs);
        self.signal_tick(inputs);

    def tickRenderInput(self,inputs):
        self.renderInput(inputs);
        self.signal_render_tick(inputs);

    @abstractmethod
    def processInput(self, inputs):
        pass;

    @abstractmethod
    def renderInput(self,inputs):
        pass;

    def close(self):
        self._close();
        self.signal_close();

    def _close(self): pass;

    def isRunning(self,data=None,useCache=False):
        if data is not None:
            return self.runConfig.gameStillRunning(data);
        elif useCache and self.mapDataCache:
            return self.runConfig.gameStillRunning(self.mapDataCache);
        return self.runConfig.gameStillRunning(self.getMappedData());

    def signal_start(self,*args,**kwargs):
        [rep.on_start(self,*args,**kwargs) for rep in self.reporters];

    def signal_training_data_load(self,id,*args,**kwargs):
        [rep.on_training_data_load(self,id,*args,**kwargs) for rep in self.reporters];

    def signal_tick(self,inputs,*args,**kwargs):
        [rep.on_tick(self,inputs,*args,**kwargs) for rep in self.reporters];

    def signal_render_tick(self,*args,**kwargs):
        [rep.on_render_tick(self,*args,**kwargs) for rep in self.reporters];

    def signal_close(self,*args,**kwargs):
        [rep.on_finish(self,*args,**kwargs) for rep in self.reporters];

    def signal_reporters(self,signal:str,*args,**kwargs):
        [rep.on_signal(self,signal,*args,**kwargs) for rep in self.reporters];



class Multi_Data_Game(RunGame):
    #Warning: will not work properly if recurrent net because the net will still have memory from the previous iteration of the game

    #kwargs requirements:
    #  -gameClass: equivalent of EvalGame gameClass, the game that should be played with multiple training data
    #  -training_data: list of all training_data to pass to the lower game
    def __init__(self,runnerConfig,kwargs):
        self.gameClass = kwargs[gameClass];
        self.runConfig = runnerConfig;
        self.data = kwargs[training_data].copy();
        self.kwargs = kwargs;
        self.data_index = 0;
        self.global_steps = 0;
        kwargs[training_data] = self.data[self.data_index];
        self.sub_game = self.gameClass(kwargs);
        self.accumulated_fitness = 0;
        self.done = False;

    def getOutputData(self):
        return self.sub_game.getOutputData();

    def isRunning(self):
        return not self.done;

    def renderInput(self):
        if (not self.done):
            result = self.sub_game.renderInput(inputs);
            if (runnerConfig.fitness_collection_type != None and runnerConfig.fitness_collection_type == 'continuous' and self.sub_game.isRunning()):
                self.accumulated_fitness += self.sub_game.getFitnessScore();
            if (not self.sub_game.isRunning()):
                self.accumulated_fitness += self.sub_game.getFitnessScore();
                self.data_index += 1;
                if (self.data_index >= len(self.data)):
                    self.done = True;
                else:
                    self.kwargs[training_data] = self.data[self.data_index];
                    self.sub_game = self.gameClass(self.kwargs);
                
            self.global_steps += 1;
            return result;
        return None;

    def processInput(self,inputs):
        if (not self.done):
            self.sub_game.processInput(inputs);
            if (runnerConfig.fitness_collection_type != None and runnerConfig.fitness_collection_type == 'continuous' and self.sub_game.isRunning()):
                self.accumulated_fitness += self.sub_game.getFitnessScore();
            if (not self.sub_game.isRunning()):
                self.accumulated_fitness += self.sub_game.getFitnessScore();
                self.data_index += 1;
                if (self.data_index >= len(self.data)):
                    self.done = True;
                else:
                    self.kwargs[training_data] = self.data[self.data_index];
                    self.sub_game = self.gameClass(self.kwargs);
                
            self.global_steps += 1;

class StarSmash(RunGame):
    def __init__(self,runnerConfig,kwargs):
        super().__init__(runnerConfig,kwargs);
        self.height = 1; #0-7 height
        self.score = 0;
        self.level = 0;
        self.asteroids=[[random.randint(1,6),15]];
        self.cooldown = 0;
        self.alive = True;
        self.firing = False;

    def renderInput(self,inputs):
        self.firing = False;
        self.processInput(inputs);
        if (not self.runConfig.gameStillRunning(self.getMappedData())):
            return Image.open('images\\epic_fail.png');
        fitness = self.runConfig.fitnessFromGameData(self.getMappedData());
        baseImage = self.get_screen(self.height,self.firing,self.asteroids,inputs);
        draw = ImageDraw.Draw(baseImage);
        font = ImageFont.truetype('arial.ttf',50);
        draw.text((5,5),'Fitness: ' + str(fitness),fill=(200,25,25));
        
        
        return baseImage;
        
        
    def processInput(self,inputs):
        self.steps += 1;
        if (inputs[2] > 0 and self.cooldown == 0):
            self.cooldown = 2;
            self.firing = True;
            self.fire();
        else:
            heightInc = -1 if inputs[0] > 0.1 else 0;
            heightInc += 1 if inputs[1] > 0.1 else 0;
            self.height += heightInc;
            self.height = 1 if self.height < 1 else (6 if self.height > 6 else self.height);
            self.cooldown = self.cooldown - 1 if self.cooldown > 0 else 0;
        self.advanceAsteroids();
        self.checkForCollisions();

    def getFirstAsteroid(self):
        firstAsteroidNum = 0;
        closestAsteroid = 15;
        for i in range(len(self.asteroids)):
            asteroid = self.asteroids[i];
            if (asteroid[1] < closestAsteroid):
                closestAsteroid = asteroid[1];
                firstAsteroidNum = i;
        resultroid = self.asteroids[firstAsteroidNum];
        return {"height":resultroid[0],"distance":resultroid[1]};

    def getSecondAsteroid(self):
        firstAsteroidNum = 0;
        closestAsteroid = 15;
        secondAsteroidNum = 0;
        secondClosestAsteroid = 15;
        for i in range(len(self.asteroids)):
            asteroid = self.asteroids[i];
            if (asteroid[1] < closestAsteroid):
                secondAsteroidNum = firstAsteroidNum;
                secondClosestAsteroid = closestAsteroid;
                closestAsteroid = asteroid[1];
                firstAsteroidNum = i;
            elif (asteroid[1] < secondClosestAsteroid):
                 secondAsteroidNum = i;
                 secondClosestAsteroid = asteroid[1];
        resultroid = self.asteroids[secondAsteroidNum];
        return {"height":resultroid[0],"distance":resultroid[1]};

    def checkForCollisions(self):
        for asteroid in self.asteroids:
            if (asteroid[1] == 0):
                if (asteroid[0] < self.height+2 and asteroid[0] > self.height-2):
                    self.alive = False;
                else:
                    asteroid[0] = random.randint(1,6);
                    asteroid[1] = 15;

    def advanceAsteroids(self):
        for asteroid in self.asteroids:
            asteroid[1] -= 1;

    def fire(self):
        for asteroid in self.asteroids:
            if (asteroid[0] == self.height):
                asteroid[0] = random.randint(1,6);
                asteroid[1] = 15;
                self.score += 1;
        if (self.score > (self.level+1)*5):
            self.level += 1;
            self.asteroids.append([random.randint(1,6),15]); 
    
    def getOutputData(self):
        return {"height":self.height,
		"score":self.score,
		"level":self.level,
		"first_asteroid_height":self.getFirstAsteroid().get("height"),
		"first_asteroid_distance":self.getFirstAsteroid().get("distance"),
		"second_asteroid_height":self.getSecondAsteroid().get("height"),
		"second_asteroid_distance":self.getSecondAsteroid().get("distance"),
		"alive":self.alive};

    

    def get_screen(self,ship_height,is_firing,asteroids,inputs):
        bg = Image.open('images\\calc_bg.png');
        self.paste_ship(bg,ship_height);
        if (is_firing):
            self.paste_beam(bg, ship_height);
        [self.paste_asteroid(bg,asteroid[1],asteroid[0]) for asteroid in asteroids];
        up = Image.open('images\\up_norm.png') if (inputs[0] <= 0) else Image.open('images\\up_press.png');
        down = Image.open('images\\down_norm.png') if (inputs[1] <= 0) else Image.open('images\\down_press.png');
        beamin = Image.open('images\\beam_norm.png') if (not(self.firing)) else Image.open('images\\beam_press.png');
    
        bg.paste(up,(855,15),up);
        bg.paste(down,(855,55),down);
        bg.paste(beamin,(900,35),beamin);
        draw = ImageDraw.Draw(bg);
        draw.text((900,15),str(inputs[2]),fill=(255 if inputs[2]>0 else 0,0,0));
        draw.text((830,30),str(inputs[0]),fill=(255 if inputs[0]>0.1 else 0,0,0));
        draw.text((830,70),str(inputs[1]),fill=(255 if inputs[1]>0.1 else 0,0,0));

        #print('inputs: {0}, {1}, {2}'.format(str(inputs[0]),str(inputs[1]),str(inputs[2])));
        
        return bg;

    def paste_asteroid(self,bg,x,y):
        asteroid_pic = Image.open('images\\asteroid.png');
        self.add_char_to_calc_grid(x,y,asteroid_pic,bg);

    def paste_beam(self,bg,height,left=2):
        beam = Image.open('images\\beam.png');
        self.add_char_to_calc_grid(left,height,beam,bg);

    def paste_ship(self,bg,height):
        self.add_char_to_calc_grid(0,height-1,Image.open('images\\ship_filled.png'),bg);

    def add_char_to_calc_grid(self,x,y,char,bg):
        bg.paste(char,(x*60,y*80),char);

