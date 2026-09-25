# Call 'python -m unittest' on this folder
from genericpath import isdir
import unittest
import sys
import os
import random


if not os.path.isdir("test"):
    sys.path.append("..")


class Test_Install(unittest.TestCase):

    def test_install(self):
        #sudo apt install python3-pip
        
        #python -m pip install -U pip

        #Install pytroch
        ##https://pytorch.org/get-started/locally/
        
        
        #python -m pip install -U matplotlib
        #pip install pytorch-lightning
        #pip install ema_pytorch
        #python -m pip install -U scikit-image
        #pip install configargparse
        #pip install einops

        
        with self.subTest(dataset='install matplotlib'):
            import matplotlib.pyplot as plt
        with self.subTest(dataset='install torch'):
            import torch
            #https://pytorch.org/get-started/locally/
        with self.subTest(dataset='install numpy'):
            import numpy as np
        with self.subTest(dataset='install pytorch-lightning'):
            import pytorch_lightning
        with self.subTest(dataset='install ema_pytorch'):        
            import ema_pytorch
        with self.subTest(dataset='install skimage'):
            import skimage
        with self.subTest(dataset='install configargparse'):
            import configargparse
        with self.subTest(dataset='install einops'):
            import einops


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):

