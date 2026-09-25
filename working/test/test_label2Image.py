# Call 'python -m unittest' on this folder
from genericpath import isdir
import unittest
import sys
import os
import random
import numpy as np


if not os.path.isdir("test"):
    sys.path.append("..")


class Test_Label_2_Image(unittest.TestCase):
    def test_dataset(self):
        from loader import load_dataset

        for train in [True, False]:
            for dataset_name in load_dataset.dataset_names_label2img:
                with self.subTest(dataset=dataset_name):
                    root = load_dataset.get_potential_roots(dataset_name, offset=1)
                    if not os.path.isdir(root) and dataset_name != "MNIST":
                        print(f"Skip dataset {dataset_name}, because it is not downloaded")
                        continue
                    size = 32
                    dataset, i = load_dataset.getDataset(
                        None, train, dataset=dataset_name, size=size, inpainting=None, offset=1
                    )
                    self.assertEqual(i, 1)
                    self.assertNotEqual(len(dataset), 0)
                    for _ in range(100):
                        idx = random.randint(0, len(dataset) - 1)
                        ds_out = dataset[idx]
                        self.assertIsInstance(ds_out, dict)
                        self.assertEqual(ds_out["target"].shape, (i, size, size))
                        self.assertIsInstance(dataset.num_classes, int)
                        self.assertLess(ds_out["label"], dataset.num_classes)

    def test_dataset_inpainting(self):
        from loader import load_dataset

        for train in [True, False]:
            for dataset_name in load_dataset.dataset_names_label2img:
                with self.subTest(dataset=dataset_name):
                    root = load_dataset.get_potential_roots(dataset_name, offset=1)
                    if not os.path.isdir(root) and dataset_name != "MNIST":
                        print(f"Skip dataset {dataset_name}, because it is not downloaded")
                        continue
                    size = 32
                    dataset, i = load_dataset.getDataset(
                        None, train, dataset=dataset_name, size=size, inpainting="random_ege", offset=1
                    )
                    self.assertEqual(i, 1)
                    self.assertNotEqual(len(dataset), 0)
                    msk = 0
                    for _ in range(250):
                        idx = random.randint(0, len(dataset) - 1)
                        ds_out = dataset[idx]
                        self.assertIsInstance(ds_out, dict)
                        self.assertEqual(ds_out["target"].shape, (i, size, size))
                        self.assertIsInstance(dataset.num_classes, int)
                        self.assertLess(ds_out["label"], dataset.num_classes)
                        if "mask" in ds_out:
                            self.assertEqual(ds_out["mask"].shape, (i, size, size))
                            msk += 1
                    self.assertNotEqual(msk, 0, "No Masks were generated")

    def test_training(self):

        import diffusion as diffusion

        opt = diffusion.get_opt("config/MNIST.conf")

        diffusion.main(opt, limit_train_batches=10)

    def tearDown(self):
        import shutil

        logs_folder = "logs_diffusion"
        if os.path.exists(logs_folder):
            for path in os.listdir(logs_folder):
                if "test_" in path:
                    path = os.path.join(logs_folder, path)
                    print("remove created folder", path)
                    shutil.rmtree(path)


if __name__ == "__main__":
    unittest.main()

# @unittest.skipIf(condition, reason)
# with self.subTest(i=i):
