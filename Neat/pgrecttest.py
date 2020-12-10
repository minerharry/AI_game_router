import pygame as pg
from py_mario_bros.PythonSuperMario_master.source.states.segmentGenerator import SegmentGenerator,GenerationOptions
#pg.init();
#SCREEN = pg.display.set_mode((400,400))

inital_config = GenerationOptions(num_blocks=[0,5],ground_height=[7,10]);
training_data = SegmentGenerator.generateBatch(inital_config,100,return_raw=True);

for datum in training_data:
    for task in datum['tasks']:
        if task in datum['blocks']:
            print(datum);
            print(task);
#print(training_data[0])

pg.display.update();
done = False;
while not done:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            done = True