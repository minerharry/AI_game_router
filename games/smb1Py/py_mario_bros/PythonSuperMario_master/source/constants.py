__author__ = 'marble_xu'

DRAW_GRID = True; #whether to draw the collision grid view on screen (for debug purposes);

NONE = 0;
LOW = 1;
MED = 2;
HIGH = 3;

GRAPHICS_SETTINGS = LOW; 
#NONE: no sprite images ever assigned nor loaded from memory; draw function never called
#LOW: all sprite images are placeholders (not loeaded from memory)
#MED: sprite images loaded from memory, but all animations deactivated
#HIGH: Original quality - all sprite images loaded from memory and fully animated

#NOTE: it is assumed that medium and high quality means that loaded maps have image backgrounds, meaning ground tiles do not render. ground will render with a placeholder on low quality
#NOTE: Certain enemies (only koopas, so far) change their hitbox between frames; specifically, when they turn into shells. All other enemies, so far, have the same hitbox independent of their animation, meaning the exact same image can be used for all frames. If any new enemies get added (and they change hitbox based on animation), the collision needs to be updated accordingly on low/medium graphics settings
#NOTE: Above issue likely irrelevant now - all unique animation frames are properly stored

DISPLAY_FRAMERATE = False;

DEBUG_RENDER = False

DEBUG = False;
DEBUG_START_X = 200
DEBUG_START_y = 538

SCREEN_HEIGHT = 600
SCREEN_WIDTH = 600
SCREEN_SIZE = (SCREEN_WIDTH,SCREEN_HEIGHT)

ORIGINAL_CAPTION = "Super Mario Bros"

## COLORS ##
#                R    G    B
GRAY         = (100, 100, 100)
NAVYBLUE     = ( 60,  60, 100)
WHITE        = (255, 255, 255)
RED          = (255,   0,   0)
GREEN        = (  0, 255,   0)
FOREST_GREEN = ( 31, 162,  35)
BLUE         = (  0,   0, 255)
SKY_BLUE     = ( 39, 145, 251)
YELLOW       = (255, 255,   0)
ORANGE       = (255, 128,   0)
PURPLE       = (255,   0, 255)
CYAN         = (  0, 255, 255)
BLACK        = (  0,   0,   0)
NEAR_BLACK   = ( 19,  15,  48)
COMBLUE      = (233, 232, 255)
GOLD         = (255, 215,   0)
BROWN        = (139,  69,  19)

PLAYER_PLACEHOLDER_COLOR = BLUE
ENEMY_PLACEHOLDER_COLOR = RED
BOX_PLACEHOLDER_COLOR = GOLD
BRICK_PLACEHOLDER_COLOR = ORANGE
GROUND_PLACEHOLDER_COLOR = BROWN
PIPE_PLACEHOLDER_COLOR = GREEN
POWERUP_PLACEHOLDER_COLOR = WHITE
SLIDER_PLACEHOLDER_COLOR = GROUND_PLACEHOLDER_COLOR

BGCOLOR = WHITE

SHADER_ALPHA = 128

SIZE_MULTIPLIER = 2.5
BRICK_SIZE_MULTIPLIER = 2.5
BACKGROUND_MULTIPLER = 2.5
PLAYER_WIDTH_MODIFIER = -0.25 #Player hitbox width is actually two pixels smaller on either side than it visually appears - 12 pixels wide instead of 16. Necessary so player can fit through 1 block gaps
GROUND_HEIGHT = SCREEN_HEIGHT - 62
TILE_SIZE = SIZE_MULTIPLIER*16


GAME_TIME_OUT = 301

#STATES FOR ENTIRE GAME
MAIN_MENU = 'main menu'
LOAD_SCREEN = 'load screen'
TIME_OUT = 'time out'
GAME_OVER = 'game over'
LEVEL = 'level'

#LEVEL GROUPS
POWERUP_GROUP = 'powerup_group'
COIN_GROUP = 'coin_group'
SHELL_GROUP = 'shell_group'
PLAYER_GROUP = 'player_group'
DYING_GROUP = 'dying_group'
GROUND_STEP_PIPE_GROUP = 'ground_step_pipe_group'
GROUND_GROUP = 'ground_group'
BOX_GROUP = 'box_group'
BRICK_GROUP = 'brick_group'
BRICKPIECE_GROUP = 'brickpiece_group'
CHECKPOINT_GROUP = 'checkpoint_group'
COIN_GROUP = 'coin_group'
ENEMY_GROUP = 'enemy_group'
FLAGPOLE_GROUP = 'flagpole_group'


#MAIN MENU CURSOR STATES
PLAYER1 = '1 PLAYER GAME'
PLAYER2 = '2 PLAYER GAME'

#GAME INFO DICTIONARY KEYS
COIN_TOTAL = 'coin total'
SCORE = 'score'
TOP_SCORE = 'top score'
LIVES = 'lives'
CURRENT_TIME = 'current time'
LEVEL_NUM = 'level num'
PLAYER_NAME = 'player name'
PLAYER_MARIO = 'mario'
PLAYER_LUIGI = 'luigi'

#MAP COMPONENTS
MAP_IMAGE = 'image_name'
MAP_BOUNDS = 'map_bounds'
MAP_START = 'map_start'
MAP_MAPS = 'maps'
MAP_FLAGX = 'flag_x'
SUB_MAP = 'sub_map'
MAP_GROUND = 'ground'
MAP_PIPE = 'pipe'
PIPE_TYPE_NONE = 0
PIPE_TYPE_IN = 1                # can go down in the pipe
PIPE_TYPE_HORIZONTAL = 2        # can go right in the pipe
MAP_STEP = 'step'
MAP_BRICK = 'brick'
BRICK_NUM = 'brick_num'
TYPE_NONE = 0
TYPE_COIN = 1
TYPE_STAR = 2
MAP_BOX = 'box'
TYPE_MUSHROOM = 3
TYPE_FIREFLOWER = 4
TYPE_FIREBALL = 5
TYPE_LIFEMUSHROOM = 6
MAP_ENEMY = 'enemy'
ENEMY_TYPE_GOOMBA = 0
ENEMY_TYPE_KOOPA = 1
ENEMY_TYPE_FLY_KOOPA = 2
ENEMY_TYPE_PIRANHA = 3
ENEMY_TYPE_FIRESTICK = 4
ENEMY_TYPE_FIRE_KOOPA = 5
ENEMY_RANGE = 'range'
MAP_CHECKPOINT = 'checkpoint'
ENEMY_GROUPID = 'enemy_groupid'
MAP_INDEX = 'map_index'
CHECKPOINT_TYPE_ENEMY = 0
CHECKPOINT_TYPE_FLAG = 1
CHECKPOINT_TYPE_CASTLE = 2
CHECKPOINT_TYPE_MUSHROOM = 3
CHECKPOINT_TYPE_PIPE = 4        # trigger player to go right in a pipe
CHECKPOINT_TYPE_PIPE_UP = 5     # trigger player to another map and go up out of a pipe
CHECKPOINT_TYPE_MAP = 6         # trigger player to go to another map
CHECKPOINT_TYPE_BOSS = 7        # defeat the boss
MAP_FLAGPOLE = 'flagpole'
FLAGPOLE_TYPE_FLAG = 0
FLAGPOLE_TYPE_POLE = 1
FLAGPOLE_TYPE_TOP = 2
MAP_SLIDER = 'slider'
HORIZONTAL = 0
VERTICAL = 1
VELOCITY = 'velocity'
MAP_COIN = 'coin'

#COMPONENT COLOR
COLOR = 'color'
COLOR_TYPE_ORANGE = 0
COLOR_TYPE_GREEN = 1
COLOR_TYPE_RED = 2

#BRICK STATES
RESTING = 'resting'
BUMPED = 'bumped'
OPENED = 'opened'

#MUSHROOM STATES
REVEAL = 'reveal'
SLIDE = 'slide'

#Player FRAMES
PLAYER_FRAMES = 'image_frames'
RIGHT_SMALL_NORMAL = 'right_small_normal'
RIGHT_BIG_NORMAL = 'right_big_normal'
RIGHT_BIG_FIRE = 'right_big_fire'

#PLAYER States
STAND = 'standing'
WALK = 'walk'
JUMP = 'jump'
FALL = 'fall'
FLY = 'fly'
SMALL_TO_BIG = 'small to big'
BIG_TO_FIRE = 'big to fire'
BIG_TO_SMALL = 'big to small'
FLAGPOLE = 'flag pole'
WALK_AUTO = 'walk auto'     # not handle key input in this state
END_OF_LEVEL_FALL = 'end of level fall'
IN_CASTLE = 'in castle'
DOWN_TO_PIPE = 'down to pipe'
UP_OUT_PIPE = 'up out of pipe'

#PLAYER FORCES
PLAYER_SPEED = 'speed'
WALK_ACCEL = 'walk_accel'
RUN_ACCEL = 'run_accel'
JUMP_VEL = 'jump_velocity'
MAX_Y_VEL = 'max_y_velocity'
MAX_RUN_SPEED = 'max_run_speed'
MAX_WALK_SPEED = 'max_walk_speed'
SMALL_TURNAROUND = .35
JUMP_GRAVITY = .31
GRAVITY = 1.01

#LIST of ENEMIES
GOOMBA = 'goomba'
KOOPA = 'koopa'
FLY_KOOPA = 'fly koopa'
FIRE_KOOPA = 'fire koopa'
FIRE = 'fire'
PIRANHA = 'piranha'
FIRESTICK = 'firestick'

#GOOMBA Stuff
LEFT = 'left'
RIGHT = 'right'
JUMPED_ON = 'jumped on'
DEATH_JUMP = 'death jump'
DEAD = 'dead';

#KOOPA STUFF
SHELL_SLIDE = 'shell slide'

#FLAG STATE
TOP_OF_POLE = 'top of pole'
SLIDE_DOWN = 'slide down'
BOTTOM_OF_POLE = 'bottom of pole'

#FIREBALL STATE
FLYING = 'flying'
BOUNCING = 'bouncing'
EXPLODING = 'exploding'

#IMAGE SHEET
ENEMY_SHEET = 'smb_enemies_sheet'
ITEM_SHEET = 'item_objects'

#SEGMENT GENERATION
EDGE = 'edge'; #on the outer ring within the task area
CENTER = 'center'; 
INNER = 'inner'; #all tiles within the task area
FLOOR = 'floor'; #tiles above a floor
AREA = 'area'; #tiles within a specified area
GROUNDED = 'grounded'; #tiles above collision
AIR = 'air'; #any unoccupied tile

#INPUT TYPES
NO_GRID = "no grid";
COLLISION_GRID = "collision grid";
ENEMY_GRID = "enemy grid";
BOX_GRID = "box grid";
BRICK_GRID = "brick grid";
POWERUP_GRID = "powerup grid";
FULL = "full";
