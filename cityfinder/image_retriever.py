from glob import glob
import os

from tqdm import tqdm
import numpy as np
from PIL import Image


class ImageComposer:
    def __init__(self, folder, tilesize=None, extension='jpg'):
        self.folder = folder
        self.tilesize = tilesize
        self.extension = extension

    def scan_folder(self, id):
        """ returns iterator to image paths """
        return glob(os.path.join(self.folder, id, '*.%s' % self.extension))

    def tile_index(self, path):
        parts = path[:-(1+len(self.extension))].split('_')  # from smth like 'foo/bar_x_y.jpg' extracts x, y
        return int(parts[-2]), int(parts[-1])

    def get_size(self, id):
        """ returns size of raster based on max{x, y} and self.tilesize """
        paths = list(self.scan_folder(id))
        max_x = np.max([self.tile_index(path)[0] for path in paths])
        max_y = np.max([self.tile_index(path)[1] for path in paths])
        return (max_x + 1) * self.tilesize, (max_y + 1) * self.tilesize

    def compose(self, id):
        w, h = self.get_size(id)
        print('w=%d, h=%d' % (w, h))
        img = np.zeros((w, h, 3), dtype=np.uint8)
        paths = list(self.scan_folder(id))
        for path in tqdm(paths, desc='composing tiles'):
            tile = Image.open(path)
            assert tile.size[0] == self.tilesize and tile.size[1] == self.tilesize
            x, y = self.tile_index(path)
            img[x * self.tilesize: x * self.tilesize + self.tilesize,
                y * self.tilesize: y * self.tilesize + self.tilesize, :] = np.asarray(tile, dtype=np.uint8)
        return Image.fromarray(img)