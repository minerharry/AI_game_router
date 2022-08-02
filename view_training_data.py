from pathlib import Path
import pickle
import numpy as np
from skimage.io import imshow

in_file = Path("data")/'gen_1530'

dat = None;
with open(in_file,'rb') as f:
    dat = pickle.load(f);

im = np.array(dat[0][0][0]['collision_grid']);

print(im)
imshow(im);
