 """Given a target image and a directory of source images, makes a photomosaic.
"""

import argparse
import cv2
import numpy as np
import os
import random
import sys

from glob import glob
from scipy.spatial import cKDTree
from skimage import color


def get_cell_tree(cell_images):
  L_vector = [np.mean(im[:,:,0]) for im in cell_images]
  a_vector = [np.mean(im[:,:,1]) for im in cell_images]
  b_vector = [np.mean(im[:,:,2]) for im in cell_images]
  return cKDTree(zip(L_vector, a_vector, b_vector))


def read_images(image_dir, size):
  extensions = ["bmp", "jpeg", "jpg", "png", "tif", "tiff", "JPEG"]
  search_paths = [os.path.join(image_dir, '*.' + ext) for ext in extensions]
  image_files = reduce(list.__add__, map(glob, search_paths))
  return [color.rgb2lab(cv2.resize(cv2.imread(f), size, interpolation=cv2.INTER_AREA))
          for f in image_files]


def main(argv=None):
  if argv is not None:
    sys.argv = argv

  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('target', help='the target image to produce')
  parser.add_argument('source', help='directory of images used to make mosaic')
  parser.add_argument('-r', '--num-rows', type=int, default=50,
                      help='number of rows of source images in final mosaic')
  parser.add_argument('-c', '--num-cols', type=int, default=100,
                      help='number of columns of source images in final mosaic')
  parser.add_argument('-d', '--repeat-distance', type=int, default=10,
                      help='minimum distance between repeating images')
  parser.add_argument('-a', '--alpha', type=float, default=0.25,
                      help='amount of cell colorization to perform')
  parser.add_argument('-o', '--output-filename',
                      default=os.path.join(os.getcwd(), 'mosaic.png'),
                      help='the output photomosaic image')
  parser.add_argument('-m', '--output-size-multiplier', type=int, default=2,
                      help='amount to multiply size of output image')
  args = parser.parse_args()

  k = 2 * (args.repeat_distance**2 + args.repeat_distance)

  target_image = cv2.imread(args.target)
  target_size = (args.output_size_multiplier * target_image.shape[1],
                 args.output_size_multiplier * target_image.shape[0])
  target_image = cv2.resize(target_image, target_size, cv2.INTER_CUBIC)

  cell_height = target_image.shape[0] / args.num_rows
  cell_width = target_image.shape[1] / args.num_cols
  cell_images = read_images(args.source, (cell_width, cell_height))
  assert len(cell_images) >= (2 * args.repeat_distance + 1)**2, \
      "Not enough images in source directory for specified repeat distance."

  cell_tree = get_cell_tree(cell_images)

  target_size = (args.num_cols * cell_width, args.num_rows * cell_height)
  target_image = cv2.resize(target_image, target_size, cv2.INTER_CUBIC)
  target_image = color.rgb2lab(target_image)
  output_image = np.zeros_like(target_image)

  num_complete = 0
  num_total = args.num_rows * args.num_cols
  used_indices = np.full((args.num_rows, args.num_cols), -1, dtype=int)
  for row in range(args.num_rows):
    i = row * cell_height
    first_row = max(row - args.repeat_distance, 0)
    for col in range(args.num_cols):
      j = col * cell_width
      first_col = max(col - args.repeat_distance, 0)
      last_col = min(col + args.repeat_distance, used_indices.shape[1] - 1)
      nearby_used_indices = used_indices[first_row:row+1,first_col:last_col+1]
      target_window = target_image[i:i+cell_height,j:j+cell_width,:]
      target_color = [np.mean(target_window[:,:,x]) for x in range(3)]
      best_index = [index for index in cell_tree.query(target_color, k=k)[1]
                    if index not in nearby_used_indices][0]
      best_match = cell_images[best_index]
      for x in range(3):
        output_image[i:i+cell_height,j:j+cell_width,x] = (
            (args.alpha * target_color[x] + (1.0 - args.alpha) * best_match[:,:,x]))
      used_indices[row,col] = best_index
      num_complete += 1
      sys.stdout.flush()
      sys.stdout.write("\r%.2f%% complete" % (100.0 * num_complete / num_total))
  print

  output_image = 255 * color.lab2rgb(output_image)
  cv2.imwrite(args.output_filename, output_image)


if __name__ == '__main__':
  sys.exit(main())
