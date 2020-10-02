import json
import random
from .segment import SegmentState
from .. import constants as c

#TODO: add sliders, enemies, and density constraints. Essentially just make it better/exist lol
class SegmentGenerator:
    def __init__(self,generationOptions):
        self.options = generationOptions;

    @staticmethod
    def generate(options,makeBatches=False):
        numTiles = options.size[0]*options.size[1];
        tiles = [(i/options.size[1],i%options.size[1]) for i in range(numTiles)];
        outputGrid = [[0 for i in range(options.size[0])] for j in range(options.size[1])];
        random.shuffle(tiles);
        innerRing = options.inner_ring();

        if (options.hasGround and options.groundHeight is not None):
            groundHeight = options.groundHeight;
            if (isinstance(groundHeight,list)):
                groundHeight = random.choice(range(groundHeight[0],groundHeight[1]+1));
            groundPositions = [[(i,j) for i in range(options.size[0])] for j in range(groundHeight,options.size[1])];
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

        if makeBatches:
            return SegmentGenerator.export(options.size,block_positions,[],[],[],player_position,[random.choice(innerRing)]);
        else:
            random.shuffle(innerRing);
            return SegmentGenerator.export(options.size,block_positions,[],[],[],player_position,innerRing[:options.taskBatchSize]);
        
        

    @staticmethod    
    def export(size,blocks,bricks,boxes,dynamics,player_start,task_positions):
        output_dict = {};
        if blocks is not None and len(blocks) > 0:
            output_dict["ground"] = [{"x":pos[0]*16,"y":pos[1]*16,"width":16,"height":16} for pos in blocks];
        if bricks is not None and len(bricks) > 0:
            print("ERROR: brick generation not done yet");
        if boxes is not None and len(boxes) > 0:
            print("ERROR: box generation not done yet");
        if dynamics is not None and len(dynamics) > 0:
            print("ERROR: dynamic object generation not done yet");
        output_dict[c.MAP_MAPS] = [{"start_x":0,"end_x":size[0]*16,"player_x":player_start[0]*16,"player_y":player_start[1]*16}];
        output_dict[c.MAP_SIZE] = [size[0]*16,size[1]*16];
        result = [];
        for pos in task_positions:
            result.append(SegmentState(None,output_dict,pos));
        return result;

        

        

    @staticmethod
    def generateBatch(options,batchSize):
        output = [];
        while len(output) < batchSize:
            output += SegmentGenerator.generate(options,makeBatches=True);
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
        inner_size = self.innerSize;
        margins = [int((size[0]-inner_size[0])/2),int(0.5+(size[0]-inner_size[0])/2),int((size[1]-inner_size[1])/2),int(0.5+(size[1]-inner_size[1])/2),] #left, right, top, bottom
        positions = [(i,margins[2]) for i in range(margins[0],size[0]-margins[1]-1)]
        positions += [(i,size[1]-margins[3]-1) for i in range(margins[0]+1,size[0]-margins[1])]
        positions += [(margins[0],i) for i in range(margins[2]+1,size[1]-margins[3])]
        positions += [(size[0] - margins[1] - 1,i) for i in range(margins[2],size[1]-margins[3]-1)]
        return positions;


