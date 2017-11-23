import unittest
import os

from cityfinder.image_retriever import ImageComposer


class TestImageRetriever(unittest.TestCase):

    base_folder = 'data/tiles'
    out_folder = 'out/composed'

    def test_compose(self):
        id = 'FL37465325' #'tulkarm'
        composer = ImageComposer(folder=self.base_folder, extension='png')
        img = composer.compose(id)
        os.makedirs(os.path.join(self.out_folder, id), exist_ok=True)
        out_path = os.path.join(self.out_folder, id, '%s_composed.png' % id)
        img.save(out_path)
