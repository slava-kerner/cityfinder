from glob import glob
import os

import numpy as np
from PIL import Image


class ImageComposer:
    def __init__(self, folder, extension='jpg'):
        self.folder = folder
        self.extension = extension

    def scan_folder(self, id):
        """ returns iterator to image paths """
        return glob(os.path.join(self.folder, id, '*.%s' % self.extension))

    def tile_index(self, path):
        parts = path[:-(1+len(self.extension))].split('_')  # from smth like 'foo/bar_x_y.jpg' extracts x, y
        return int(parts[-2]), int(parts[-1])

    def get_size(self, id, tilesize):
        """ returns size of raster based on max{x, y} and tilesize """
        paths = list(self.scan_folder(id))
        max_x = np.max([self.tile_index(path)[0] for path in paths])
        max_y = np.max([self.tile_index(path)[1] for path in paths])
        return max_x + tilesize, max_y + tilesize

    def compose(self, id):
        paths = list(self.scan_folder(id))
        tile_w, tile_h = Image.open(paths[0]).size
        assert tile_w == tile_h
        tilesize = tile_w
        
        image_w, image_h = self.get_size(id, tilesize)
        print('w=%d, h=%d' % (image_w, image_h))
        img = np.zeros((image_w, image_h, 3), dtype=np.uint8)
        for path in paths:
            tile = Image.open(path)
            if tile.size[0] != tilesize or tile.size[1] != tilesize:
                continue
            # assert tile.size[0] == tilesize and tile.size[1] == tilesize
            x, y = self.tile_index(path)
            img[x: x + tilesize, y: y + tilesize, :] = np.asarray(tile, dtype=np.uint8).transpose((1, 0, 2))
        return Image.fromarray(img.transpose((1, 0, 2)))