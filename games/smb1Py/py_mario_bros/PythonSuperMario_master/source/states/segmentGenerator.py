from __future__ import annotations
import json
import random
from .segment import SegmentState
from .. import constants as c
from ..components.enemy import create_enemy

#TODO: add sliders, enemies, and density constraints. Essentially just make it better/exist lol
class SegmentGenerator:
    
    #TODO: Have different lists for each type of purpose - ex: player start, task position, enemy placement, etc.
    @staticmethod
    def generate(options:GenerationOptions,makeBatches=False,return_raw=False):
        numTiles = options.size[0]*options.size[1];
        tiles = [(int(i/options.size[1]),i%options.size[1]) for i in range(numTiles)];
        numInner = options.innerSize[0]*options.innerSize[1]
        margins = options.inner_margins();
        innerTiles = [(int(i/options.innerSize[1])+margins[0],i%options.innerSize[1]+margins[2]) for i in range(numInner)];
        innerRing = options.inner_ring();



        groundPositions = [];
        floorPositions = [];
        if (options.hasGround and options.groundHeight is not None):
            groundHeight = options.groundHeight;
            if (isinstance(groundHeight,list)):
                groundHeight = random.choice(range(groundHeight[0],groundHeight[1]+1));
            groundPositions = sum([[(i,j) for i in range(options.size[0])] for j in range(groundHeight,options.size[1])],[]);
            [innerRing.remove(el) for el in groundPositions if el in innerRing];
            [tiles.remove(el) for el in groundPositions if el in tiles];
            [innerTiles.remove(el) for el in groundPositions if el in innerTiles];
            floorPositions = [(i,groundHeight-1) for i in range(options.inner_margins()[0],options.size[0]-options.inner_margins()[1])];


        player_position = None;
        if options.startBlocks == c.INNER:
            player_position = random.choice(innerTiles);
        elif options.startBlocks == c.EDGE:
            player_position = random.choice(innerRing);
        elif options.startBlocks == c.FLOOR:
            player_position = random.choice(floorPositions);


        if player_position in tiles:
            tiles.remove(player_position)
        if player_position in innerRing:
            innerRing.remove(player_position);
        if player_position in innerTiles:
            innerTiles.remove(player_position);
        if player_position in floorPositions:
            floorPositions.remove(player_position);
        numBlocks = options.numBlocks
        if (isinstance(options.numBlocks,list)):
            numBlocks = random.choice(range(numBlocks[0],numBlocks[1]+1));
        
        block_positions = random.sample(tiles,numBlocks);
        block_positions += groundPositions;

        [innerRing.remove(pos) for pos in block_positions if pos in innerRing];

        enemies = [];

        if options.num_enemies is not None and len(options.num_enemies) > 0:
            #print(options.num_enemies)
            for enemy_type,num in options.num_enemies.items():
                if isinstance(num,list):
                    num = random.choice(range(num[0],num[1]+1));
                for i in range(num):
                    pos = random.choice(tiles);
                    tiles.remove(pos);
                    direction = random.choice([0,1]);
                    enemies.append([enemy_type,pos,direction])

        bounds = options.inner_margins();
        bounds[1] = options.size[0]-bounds[1];
        bounds[3] = options.size[1]-bounds[3];


        dynamics = {'enemies':enemies};
        batchSize = 1;
        if makeBatches:
            if isinstance(options.taskBatchSize,list):
                batchSize = random.choice(range(options.taskBatchSize[0],options.taskBatchSize[1]+1));
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

        raw_data = [options.size,block_positions,[],[],dynamics,player_position, random.sample(task_options,min(len(task_options),batchSize)),bounds];
        if return_raw:
            return [{k:v for k,v in zip(['size','blocks','bricks','boxes','dynamics','start','tasks','bounds'],raw_data)}];
        return SegmentGenerator.export(*raw_data);
        
        

    @staticmethod    
    def export(size,blocks,bricks,boxes,dynamics,player_start,task_positions,task_bounds):
        output_statics = {};
        output_dynamics = {};
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
                #print('doing_enemies')
                #print(output_dynamics)
            else:
                print("ERROR: non-enemy dynamic object generation not done yet");
        output_statics[c.MAP_MAPS] = [{c.MAP_BOUNDS:[0,size[0]*c.TILE_SIZE,0,size[1]*c.TILE_SIZE],c.MAP_START:[(player_start[0] + 0.5)*c.TILE_SIZE,(player_start[1] + 1)*c.TILE_SIZE]}]; #Add 1 to y and 0.5 to x because player map start is bottom middle not top left
        result = [];
        scaled_bounds = [bound*c.TILE_SIZE for bound in task_bounds]
        #print(task_bounds);
        #print(player_start);
        #print(scaled_bounds);
        # print(task_positions)
        for pos in task_positions:
            pos = [(i + 0.5) * c.TILE_SIZE for i in pos];
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


    def __init__(self,size=[13,13],inner_size=[7,7],has_ground=True,num_blocks=0,num_enemies={},enemy_options={},valid_task_blocks = c.INNER, valid_start_blocks = c.INNER, valid_task_positions = c.CENTER,task_batch_size = 3,ground_height = 2):
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


