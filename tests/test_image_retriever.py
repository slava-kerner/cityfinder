import unittest
import os

from cityfinder.image_retriever import ImageComposer


class TestImageRetriever(unittest.TestCase):

    base_folder = 'data/tiles'
    out_folder = 'out/composed'

    def test_compose(self):
        id = 'tulkarm'
        composer = ImageComposer(folder=self.base_folder, tilesize=200)
        paths = list(composer.scan_folder(id))
        x, y = composer.tile_index(paths[1])
        # print(paths[1], x, y)
        # print(composer.get_size(id))
        img = composer.compose(id)
        os.makedirs(os.path.join(self.out_folder, id))
        out_path = os.path.join(self.out_folder, id, '%s_composed.png' % id)
        img.save(out_path)
