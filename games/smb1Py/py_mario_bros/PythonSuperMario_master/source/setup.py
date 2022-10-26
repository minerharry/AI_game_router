__author__ = 'marble_xu'
import os
from pathlib import Path
from . import constants as c
import pygame as pg

#TO ANY WHO WONDER WHY THIS EXISTS: There used to be more pygame stuff in this file. Really stupid stuff. It initialized all the pygame variables - unnecessarily - upon import. I have removed that, and this is the only thing left.

_GFX = None;
def get_GFX():
    global _GFX
    if _GFX is None:
        pg.init()
        pg.event.set_allowed([pg.KEYDOWN, pg.KEYUP, pg.QUIT])
        pg.display.set_caption(c.ORIGINAL_CAPTION)
        pg.display.set_mode(c.SCREEN_SIZE)
        _GFX = load_all_gfx(Path("games\\smb1Py\\py_mario_bros\\PythonSuperMario_master\\resources\\graphics"))
        return _GFX
    else:
        return _GFX
        

def load_all_gfx(directory:Path, colorkey=(255,0,255), accept=('.png', '.jpg', '.bmp', '.gif')):
    print(os.getcwd());
    graphics = {}
    for pic in os.listdir(directory):
        name, ext = os.path.splitext(pic)
        if c.GRAPHICS_SETTINGS >= c.MED or (name.startswith('level') and c.GRAPHICS_SETTINGS != c.NONE):
            if ext.lower() in accept:
                img = pg.image.load(os.path.join(directory, pic))
                if img.get_alpha():
                    img = img.convert_alpha()
                else:
                    img = img.convert()
                    img.set_colorkey(colorkey)
                graphics[name] = img
        else:
            graphics[name] = None;
    return graphics
