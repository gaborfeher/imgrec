#!/usr/bin/env python3

# Crops and resamples an image to have the size of 32x32 pixel. Cropping is always
# from the middle.

import subprocess
import sys

cmd = ['identify', sys.argv[1]]
print(cmd)
img_desc = subprocess.check_output(cmd)
width, height = img_desc.decode('utf-8').split(' ')[2].split('x')
width = int(width)
height = int(height)
new_size = min(width, height)

crop_x = int((width - new_size) / 2)
crop_y = int((height - new_size) / 2)

cmd = [
    'convert',
    sys.argv[1],
    '-crop', '{}x{}+{}+{}'.format(new_size, new_size, crop_x, crop_y),
    '-sample', '32x32',
    '-colorspace', 'sRGB', '-colors', '16777216',  # try to get 3x8 bits
    sys.argv[2]]
print(cmd)
subprocess.check_call(cmd)

