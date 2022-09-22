# from cmath import rect
import itertools
import pstats
from time import time
import pygame as pg
from random import random as r
import numpy as np
# from skimage.io import imshow,show;
import cProfile as profile

centers = None;
rectarray = None;
last_offset = None;
def test(collision_type):
    global centers;
    global rectarray;
    global last_offset;
    scale = 200
    offset = [r()*scale/2,r()*scale/2];
    rects = [pg.rect.Rect(r()*scale+offset[0],r()*scale+offset[1],r()*scale,r()*scale) for i in range(20)];

    # print(rects);
    ndivisions = 50;

    size = 2*scale,2*scale;

    pieces = size[0]/ndivisions,size[1]/ndivisions;

    if collision_type == 'centers':
        if centers is None:
            centers = [[((n+0.5)*pieces[0],(j+0.5)*pieces[1]) for n in range(ndivisions)] for j in range(ndivisions)];
        else:
            diff = (offset[0]-last_offset[0],offset[1]-last_offset[1]);
            centers = [[(c[0]+diff[0],c[1]+diff[1]) for c in row] for row in centers];
        np.array([[any(r.collidepoint(centers[i][j]) for r in rects) for i in range(ndivisions)] for j in range(ndivisions)]);

    if collision_type == 'rects':
        if rectarray is None:
            rectarray = [[pg.rect.Rect(n*pieces[0],j*pieces[1],pieces[0],pieces[1]) for n in range(ndivisions)] for j in range(ndivisions)];
        else:
            diff = (offset[0]-last_offset[0],offset[1]-last_offset[1]);
            rectarray = [[re.move(diff) for re in row] for row in rectarray];
        np.array([[rectarray[i][j].collidelist(rects) != -1 for i in range(ndivisions)] for j in range(ndivisions)]);

    last_offset = offset;

def test_batch(num_iterations,collision_type):
    for i in range(num_iterations):
        test(collision_type);

def test_time(total_time,collision_type):
    end = time() + total_time;
    while (time() < end):
        test(collision_type);

# def iter_print():
#     print("a");
#     yield 0;
#     print("b");
#     yield 0;
#     print("c");
#     yield 1;
#     print("d");
#     yield 0;
#     print("e");
#     yield 1;
#     print("f");
#     yield 1;
#     print("g");
#     yield 0;


if __name__ == "__main__":
    for type in ['centers','rects']:
        start = time();
        test_batch(1000,type)
        elapsed = time() - start;
        print(f"elapsed time for {type}: {elapsed}");

    for type in ['centers','rects']:
        n = type+'_profile.stats'
        profile.run(f'test_time(5,type)',n);
        stats = pstats.Stats(n);
        print("========== Stats for",type,"==========");
        stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE).print_stats();


