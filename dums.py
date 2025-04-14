from skimage.io import imread,imsave
from skimage.exposure import rescale_intensity

im = imread("random2_s16_t109.TIF");
im = rescale_intensity(im);
imsave("phase_example.png",im);