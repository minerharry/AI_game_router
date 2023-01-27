import itertools
from os import PathLike
from pathlib import Path
import numpy as np
import games.smb1Py.py_mario_bros.PythonSuperMario_master.source.constants as constants
from games.smb1Py.py_mario_bros.PythonSuperMario_master.source.states.segment import SegmentState
from rectangle_simplification import simplify_rectangles

#TODO: add support for missing items: springs, enemies, pipe textures, flagpoles, bridges, sliders, teeters, bullet bills
def text_to_level(text:list[list[str]]|np.ndarray,blockscale:int): #text is assumed to be indexed [x,y]
    pt = tuple[int,int]
    ptt = tuple[pt,str]
    startpos = None;
    grounds:list[pt] = [];
    blocks = [];
    boxes = [];
    bricks:list[ptt] = [];
    enemies:list[ptt] = [];
    pipeCollision:list[pt] = [];
    springs = [];
    mushrooms = [];
    sliders = [];
    coins = [];
    teeters = [];
    bridges = [];
    bullets = [];
    flagpole = None;
    for x,col in enumerate(text):
        for y,c in enumerate(col[::-1]):
            p = (x,y);
            match c:
                case '.':
                    pass;
                case 'A':
                    if startpos is None:
                        startpos = p;
                case '=':
                    grounds.append(p);
                case 'b':
                    bricks.append((p,"coin"));
                case 'g':
                    enemies.append((p,"goomba"));
                case 'k':
                    enemies.append((p,"koopa"));
                case 'K':
                    enemies.append((p,"flykoopa"));
                case 'H':
                    enemies.append((p,"hammerbro"));
                case '8':
                    bullets.append(p);
                case '-':
                    blocks.append(p);
                case 'p':
                    pipeCollision.append(p);
                case '!':
                    boxes.append((p,"mushroom"))
                case '?':
                    boxes.append((p,"coin"));
                case 'F':
                    if flagpole is None:
                        flagpole = p;
                case 'x':
                    springs.append(p);
                case 'I':
                    mushrooms.append(p);
                case '0':
                    coins.append(p);
                case '<':
                    sliders.append((p,'down'));
                case '^':
                    sliders.append((p,'left'));
                case '>':
                    sliders.append((p,'up'));
                case 'v':
                    teeters.append(p);
                case '_':
                    pass;
                case '|':
                    bridges.append(p);
                case a:
                    raise Exception(f"unknown symbol {a}, {c}");
    if not startpos:
        raise Exception();

    output_statics = {};
    output_dynamics = {};
    output_statics[constants.MAP_GROUND] = [{"x":rect.left*blockscale,"y":rect.top*blockscale,"width":rect.width*blockscale,"height":rect.height*blockscale} for rect in simplify_rectangles(grounds)];
    output_statics[constants.MAP_BRICK] = [{"x":x*blockscale,"y":y*blockscale,"type":{"coin":constants.TYPE_COIN}[t]} for (x,y),t in bricks]
    output_statics[constants.MAP_STEP] = [{"x":rect.left*blockscale,"y":rect.top*blockscale,"width":rect.width*blockscale,"height":rect.height*blockscale} for rect in simplify_rectangles(blocks)];
    output_statics[constants.MAP_BOX] = [{"x":x*blockscale,"y":y*blockscale,"type":{"mushroom":constants.TYPE_MUSHROOM,"coin":constants.TYPE_COIN}[t]} for (x,y),t in boxes];
    
    ## TODO: PIPES
    output_statics[constants.MAP_GROUND] += [{"x":rect.left*blockscale,"y":rect.top*blockscale,"width":rect.width*blockscale,"height":rect.height*blockscale} for rect in simplify_rectangles(pipeCollision)];

    size = (len(text),len(text[0]));
    output_statics[constants.MAP_MAPS] = [{constants.MAP_BOUNDS:[0,size[0]*blockscale,0,size[1]*blockscale],constants.MAP_START:((startpos[0] + 0.5)*blockscale,(startpos[1] + 0.5)*blockscale)}];

    if flagpole:
        output_statics[constants.MAP_MAPS][0][constants.MAP_FLAGX] = flagpole[0]*blockscale;

    level = SegmentState(output_dynamics,output_statics);
    return level

    # print(startpos,flagpole,grounds,blocks,boxes,bricks,enemies,pipeCollision,springs);

def level_from_txt_file(file:PathLike):
    with open(file,'r') as f:
        lines = np.array([a.strip('\n') for a in f.readlines()]);
    level = text_to_level(lines,int(constants.TILE_SIZE));
    return level;

if __name__ == "__main__":
    path = "levels/smb1_levels/txt/{0}-{1}.level";
    for w,l in itertools.product(range(1,9),range(1,5)):
        p = Path(path.format(w,l));
        if p.exists():
            try:
                level = (level_from_txt_file(p));
                # print(l);
                # print(l.static_data);
                level.save_file(f"levels/smb1_levels/{w}-{l}.lvl");

            except:
                raise Exception(f"error parsing from level {w}-{l}");