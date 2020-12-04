import json
import random
from .segment import SegmentState
from .. import constants as c

#TODO: add sliders, enemies, and density constraints. Essentially just make it better/exist lol
class SegmentGenerator:
    
    @staticmethod
    def generate(options,makeBatches=False,return_raw=False):
        numTiles = options.size[0]*options.size[1];
        tiles = [(int(i/options.size[1]),i%options.size[1]) for i in range(numTiles)];
        outputGrid = [[0 for i in range(options.size[0])] for j in range(options.size[1])];
        random.shuffle(tiles);
        innerRing = options.inner_ring();

        groundPositions = [];
        if (options.hasGround and options.groundHeight is not None):
            groundHeight = options.groundHeight;
            if (isinstance(groundHeight,list)):
                groundHeight = random.choice(range(groundHeight[0],groundHeight[1]+1));
            groundPositions = sum([[(i,j) for i in range(options.size[0])] for j in range(groundHeight,options.size[1])],[]);
            [[innerRing.remove(el) for el in pos if el in innerRing] for pos in groundPositions];
            [[tiles.remove(el) for el in pos if el in tiles] for pos in groundPositions];
        


        player_position = random.choice(innerRing);
        if player_position in tiles:
             tiles.remove(player_position)
        if player_position in innerRing:
            innerRing.remove(player_position);
        numBlocks = options.numBlocks
        if (isinstance(options.numBlocks,list)):
            numBlocks = random.choice(range(numBlocks[0],numBlocks[1]+1));
            
        block_positions = random.sample(tiles,numBlocks);
        block_positions += groundPositions;

        [innerRing.remove(pos) for pos in block_positions if pos in innerRing];


        bounds = options.inner_margins();
        bounds[1] = options.size[0]-bounds[1];
        bounds[3] = options.size[1]-bounds[3];

        if makeBatches:
            random.shuffle(innerRing);
        raw_data = [options.size,block_positions,[],[],[],player_position,[random.choice(innerRing)] if not makeBatches else innerRing[:options.taskBatchSize],bounds]
        if return_raw:
            return [{k:v for k,v in zip(['size','blocks','bricks','boxes','dynamics','start','tasks','bounds'],raw_data)}];
        return SegmentGenerator.export(*raw_data);
        
        

    @staticmethod    
    def export(size,blocks,bricks,boxes,dynamics,player_start,task_positions,task_bounds):
        output_dict = {};
        if blocks is not None and len(blocks) > 0:
            output_dict["ground"] = [{"x":pos[0]*c.TILE_SIZE,"y":pos[1]*c.TILE_SIZE,"width":c.TILE_SIZE,"height":c.TILE_SIZE} for pos in blocks];
        if bricks is not None and len(bricks) > 0:
            print("ERROR: brick generation not done yet");
        if boxes is not None and len(boxes) > 0:
            print("ERROR: box generation not done yet");
        if dynamics is not None and len(dynamics) > 0:
            print("ERROR: dynamic object generation not done yet");
        output_dict[c.MAP_MAPS] = [{c.MAP_BOUNDS:[0,size[0]*c.TILE_SIZE,0,size[1]*c.TILE_SIZE],c.MAP_START:[player_start[0]*c.TILE_SIZE,player_start[1]*c.TILE_SIZE]}];
        result = [];
        scaled_bounds = [bound*c.TILE_SIZE for bound in task_bounds]
        #print(task_bounds);
        #print(scaled_bounds);
        for pos in task_positions:
            pos = [(i + 0.5) * c.TILE_SIZE for i in pos];
            result.append(SegmentState(None,output_dict,task=pos,task_bounds=scaled_bounds));
        return result;

        

        

    @staticmethod
    def generateBatch(options,batchSize,**kwargs):
        output = [];
        while len(output) < batchSize:
            output += SegmentGenerator.generate(options,makeBatches=True,**kwargs);
        return output[:batchSize];


#work in progress; new variables and customization to be added soon
class GenerationOptions:
    def __init__(self,size=[13,13],inner_size=[7,7],has_ground=True,num_blocks=0,valid_task_blocks = c.EDGE, valid_start_blocks = c.EDGE, valid_task_positions = c.CENTER,task_batch_size = 4,ground_height = 2):
        self.size = size;
        self.innerSize = inner_size;
        self.hasGround = has_ground;
        self.numBlocks = num_blocks;
        self.taskBlocks = valid_task_blocks;
        self.startBlocks = valid_start_blocks;
        self.taskPositions = valid_task_positions;
        self.taskBatchSize = task_batch_size;
        self.groundHeight = ground_height;

    def inner_ring(self):
        size = self.size;
        margins = self.inner_margins();
        positions = [(i,margins[2]) for i in range(margins[0],size[0]-margins[1]-1)]
        positions += [(i,size[1]-margins[3]-1) for i in range(margins[0]+1,size[0]-margins[1])]
        positions += [(margins[0],i) for i in range(margins[2]+1,size[1]-margins[3])]
        positions += [(size[0] - margins[1] - 1,i) for i in range(margins[2],size[1]-margins[3]-1)]
        return positions;

    def inner_margins(self):
        size = self.size;
        inner_size = self.innerSize;
        return [int((size[0]-inner_size[0])/2),int(0.5+(size[0]-inner_size[0])/2),int((size[1]-inner_size[1])/2),int(0.5+(size[1]-inner_size[1])/2)] #left, right, top, bottom



if __name__ == "__main__":
    options = GenerationOptions();
    print(options.inner_ring());


