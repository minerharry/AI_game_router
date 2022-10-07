from __future__ import annotations
import json
import random
from typing import Any, Literal
from .segment import SegmentState
from .. import constants as c
from ..components.enemy import create_enemy

def dist(p1:tuple[int,int],p2:tuple[int,int]):
    return ((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)**1/2;

#TODO: add sliders, enemies, and density constraints. Essentially just make it better/exist lol
class SegmentGenerator:
    
    #TODO: Have different lists for each type of purpose - ex: player start, task position, enemy placement, etc.
    @staticmethod
    def generate(options:GenerationOptions,makeBatches=False,return_raw=False):
        numTiles = options.size[0]*options.size[1];
        tiles = [(int(i/options.size[1]),int(i%options.size[1])) for i in range(numTiles)];
        numInner = options.innerSize[0]*options.innerSize[1]
        margins = options.inner_margins();
        innerTiles = [(int(i/options.innerSize[1])+margins[0],i%options.innerSize[1]+margins[2]) for i in range(numInner)];
        innerRing = options.inner_ring();


        groundPositions = [];
        floorPositions = [];
        if (options.hasGround and options.groundHeight is not None):
            groundHeight = options.groundHeight;
            if (isinstance(groundHeight,tuple)):
                groundHeight = random.choice(range(groundHeight[0],groundHeight[1]+1));
            groundPositions = sum([[(i,j) for i in range(options.size[0])] for j in range(groundHeight,options.size[1])],[]);
            [innerRing.remove(el) for el in groundPositions if el in innerRing];
            [tiles.remove(el) for el in groundPositions if el in tiles];
            [innerTiles.remove(el) for el in groundPositions if el in innerTiles];
            floorPositions = [(i,groundHeight-1) for i in range(options.inner_margins()[0],options.size[0]-options.inner_margins()[1])];

        tile_dict = {c.INNER:innerTiles,c.EDGE:innerRing,c.FLOOR:floorPositions,c.AIR:tiles};

        player_position = None;
        if isinstance(options.startBlocks,tuple):
            player_position = options.startBlocks;
        else:
            try:
                player_position = random.choice(tile_dict[options.startBlocks]);
            except:
                # print(options.startBlocks);
                # print(tile_dict[options.startBlocks]);
                # print(innerTiles);
                raise Exception("player position must be possible");


        if player_position in tiles:
            tiles.remove(player_position)
        if player_position in innerRing:
            innerRing.remove(player_position);
        if player_position in innerTiles:
            innerTiles.remove(player_position);
        if player_position in floorPositions:
            floorPositions.remove(player_position);

        numBlocks = options.numBlocks
        if (isinstance(numBlocks,tuple)):
            numBlocks = random.choice(range(numBlocks[0],numBlocks[1]+1));
        
        grounded = floorPositions.copy();
        block_positions = random.sample(tiles,numBlocks);
        block_positions += groundPositions;
        for pos in block_positions:
            if pos in innerRing: innerRing.remove(pos)
            if pos in innerTiles: innerTiles.remove(pos);
            if pos in floorPositions: floorPositions.remove(pos);
            if pos in tiles: tiles.remove(pos);
            if pos in grounded: grounded.remove(pos);
            if (pos[0],pos[1]-1) in tiles:
                grounded.append((pos[0],pos[1]-1));
        
        tile_dict[c.GROUNDED] = grounded;

        enemies = [];
        if options.num_enemies is not None and len(options.num_enemies) > 0:
            dicted_tiles = isinstance(options.valid_enemy_positions,dict);
            valid_tiles = tile_dict[options.valid_enemy_positions] if not dicted_tiles else None;
            for enemy_type,num in options.num_enemies.items():
                if (dicted_tiles):
                    valid_tiles = tile_dict[options.valid_enemy_positions[enemy_type]];
                if isinstance(num,tuple):
                    num = random.choice(range(num[0],num[1]+1));
                for _ in range(num):
                    pos = random.choice(valid_tiles);
                    tiles.remove(pos);
                    direction = random.choice([0,1]);
                    enemies.append([enemy_type,pos,direction])


        if options.hasGround:

            available_xs = list(range(options.inner_margins()[0],options.size[0]-options.inner_margins()[1]));
            if not(options.allow_gap_under_start):
                if (player_position[0] in available_xs):
                    available_xs.remove(player_position[0]);
            
            num_gaps = options.num_gaps;
            if (isinstance(num_gaps,tuple)):
                num_gaps = random.randint(*num_gaps)

            gaps = [];    
            for _ in range(num_gaps):
                width = options.gap_width;
                if (isinstance(width,tuple)):
                    width = random.randint(*width);
                if width > len(available_xs):
                    break;
                gap_start = random.randint(0,len(available_xs)-width);
                gap = available_xs[gap_start:gap_start+width];
                [available_xs.remove(g) for g in gap];
                gaps += gap;
            
            [[block_positions.remove((x,y)) for y in range(groundHeight,options.size[1])] for x in gaps]    


        bounds = options.inner_margins();
        bounds[1] = options.size[0]-bounds[1];
        bounds[3] = options.size[1]-bounds[3];
        bounds[0] -= options.bounds_offset;
        bounds[2] -= options.bounds_offset;
        bounds[1] += options.bounds_offset;
        bounds[3] += options.bounds_offset;

        dynamics = {'enemies':enemies};

        batchSize = 1;
        if makeBatches:
            if isinstance(options.taskBatchSize,tuple):
                batchSize = random.randint(*options.taskBatchSize);
            else:
                batchSize = options.taskBatchSize;

        task_options = [];
        # print(options.taskBlocks)
        if options.taskBlocks == c.EDGE:
            task_options = innerRing;
        elif options.taskBlocks == c.FLOOR:
            task_options = floorPositions;
        elif options.taskBlocks == c.INNER:
            task_options = innerTiles;

        if options.task_dist_range is not None:
            if options.task_dist_range[0] is not None:
                task_options = [t for t in task_options if dist(player_position,t) > options.task_dist_range[0]]
            if options.task_dist_range[1] is not None:
                task_options = [t for t in task_options if dist(player_position,t) < options.task_dist_range[0]]

        
        print(task_options);

        raw_data = [options.size,block_positions,[],[],dynamics,player_position, random.sample(task_options,min(len(task_options),batchSize)),bounds];
        if return_raw:
            return [{k:v for k,v in zip(['size','blocks','bricks','boxes','dynamics','start','tasks','bounds'],raw_data)}];
        return SegmentGenerator.export(*raw_data);
        
        

    @staticmethod    
    def export(size:tuple[int,int],blocks:list[tuple[int,int]],bricks:list,boxes:list,dynamics:dict[str,Any],player_start:tuple[int,int],task_positions:list[tuple[int,int]],task_bounds:tuple[int,int]):
        output_statics = {};
        output_dynamics = {};
        print(task_positions);
        if blocks is not None and len(blocks) > 0:
            output_statics["ground"] = [{"x":pos[0]*c.TILE_SIZE,"y":pos[1]*c.TILE_SIZE,"width":c.TILE_SIZE,"height":c.TILE_SIZE} for pos in blocks];
        if bricks is not None and len(bricks) > 0:
            print("ERROR: brick generation not done yet");
        if boxes is not None and len(boxes) > 0:
            print("ERROR: box generation not done yet");
        if dynamics is not None and len(dynamics) > 0:
            if 'enemies' in dynamics:
                enemies = dynamics['enemies'];
                enemy_output = [];
                for enemy_dat in enemies:
                    item = {"x":enemy_dat[1][0]*c.TILE_SIZE,"y":enemy_dat[1][1]*c.TILE_SIZE,"type":enemy_dat[0],"direction":enemy_dat[2],"color":0}
                    enemy_output.append(item);
                output_statics["enemy"]={"-1":enemy_output};
            else:
                print("ERROR: non-enemy dynamic object generation not done yet");
        output_statics[c.MAP_MAPS] = [{c.MAP_BOUNDS:[0,size[0]*c.TILE_SIZE,0,size[1]*c.TILE_SIZE],c.MAP_START:((player_start[0] + 0.5)*c.TILE_SIZE,(player_start[1] + 1)*c.TILE_SIZE)}]; #Add 1 to y and 0.5 to x because player map start is bottom middle not top left
        result:list[SegmentState] = [];
        scaled_bounds = [bound*c.TILE_SIZE for bound in task_bounds]
        #print(task_bounds);
        #print(player_start);
        #print(scaled_bounds);
        # print(task_positions)
        for pos in task_positions:
            pos = tuple([(i + 0.5) * c.TILE_SIZE for i in pos]);
            result.append(SegmentState(output_dynamics,output_statics,task=pos,task_bounds=scaled_bounds));
        #print(result[0].dynamic_data)
        return result;

        

        

    @staticmethod
    def generateBatch(options,batchSize,**kwargs):
        output = [];
        print(batchSize)
        while len(output) < batchSize:
            output += SegmentGenerator.generate(options,makeBatches=True,**kwargs);
            # print(output);
            print(len(output));
        return output[:batchSize];


#work in progress; new variables and customization to be added soon
class GenerationOptions:
    enemy_list = [c.ENEMY_TYPE_GOOMBA,c.ENEMY_TYPE_KOOPA,c.ENEMY_TYPE_FLY_KOOPA,c.ENEMY_TYPE_PIRANHA,c.ENEMY_TYPE_FIRE_KOOPA,c.ENEMY_TYPE_FIRESTICK]


    def __init__(self,
    size:tuple[int,int]=(17,13),
    inner_size:tuple[int,int]=(11,7),
    has_ground=True,
    num_blocks:int|tuple[int,int]=0,
    num_enemies:dict[int,int|tuple[int,int]]={},
    valid_enemy_positions:str|dict[int,str]=c.AIR,
    enemy_options:dict[int,Any]={},
    valid_task_blocks = c.INNER,
    valid_start_blocks:str|tuple[int,int] = c.INNER,
    valid_task_positions = c.CENTER,
    task_batch_size:int|tuple[int,int] = 3,
    ground_height:int|tuple[int,int] = 7, #top down
    num_gaps:int|tuple[int,int] = 0,
    gap_width:int|tuple[int,int] = 0,
    allow_gap_under_start=False,
    task_distance_range:None|tuple[float|None,float|None]=(2,None),
    bounds_offset:int=2. #how much larger the play area should be than the "bounds" of the segment; ensures space for player to maneuver and not die
    ):

        self.size = size;
        self.innerSize = inner_size;
        self.hasGround = has_ground;
        self.numBlocks = num_blocks;
        self.taskBlocks = valid_task_blocks;
        self.startBlocks = valid_start_blocks;
        self.taskPositions = valid_task_positions;
        self.taskBatchSize = task_batch_size;
        self.groundHeight = ground_height;
        self.num_enemies = num_enemies;
        self.enemy_options = enemy_options;
        self.valid_enemy_positions = valid_enemy_positions;
        self.num_gaps = num_gaps;
        self.gap_width = gap_width;
        self.allow_gap_under_start = allow_gap_under_start;
        self.task_dist_range = task_distance_range;
        self.bounds_offset = bounds_offset;

        self._margins = None;
        

    def inner_ring(self):
        size = self.size;
        margins = self.inner_margins();
        positions = [(i,margins[2]) for i in range(margins[0],size[0]-margins[1]-1)]
        positions += [(i,size[1]-margins[3]-1) for i in range(margins[0]+1,size[0]-margins[1])]
        positions += [(margins[0],i) for i in range(margins[2]+1,size[1]-margins[3])]
        positions += [(size[0] - margins[1] - 1,i) for i in range(margins[2],size[1]-margins[3]-1)]
        return positions;

    def inner_margins(self):
        if self._margins is None:
            size = self.size;
            inner_size = self.innerSize;
            self._margins = [int((size[0]-inner_size[0])/2),int(0.5+(size[0]-inner_size[0])/2),int((size[1]-inner_size[1])/2),int(0.5+(size[1]-inner_size[1])/2)] #left, right, top, bottom
        return self._margins.copy();



if __name__ == "__main__":
    options = GenerationOptions();


