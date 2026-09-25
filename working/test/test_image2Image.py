# Call 'python -m unittest' on this folder
from genericpath import isdir
import unittest
import sys
import os
import random


if not os.path.isdir("test"):
    sys.path.append("..")


class Test_Image2Image(unittest.TestCase):
    def test_dataset(self):
        from loader import load_dataset

        num_ds = 0
        for train in [True, False]:
            for dataset_name in load_dataset.dataset_names_img2img:
                with self.subTest(dataset=dataset_name):
                    root = load_dataset.get_potential_roots(dataset_name, offset=1)
                    if not os.path.isdir(root):
                        print(f"Skip dataset {dataset_name}, because it is not downloaded")
                        continue
                    num_ds += 1
                    size = 32
                    dataset, i = load_dataset.getDataset(None, train, dataset=dataset_name, size=size, offset=1)
                    self.assertEqual(i, 3)

                    self.assertNotEqual(len(dataset), 0)
                    for _ in range(100):
                        idx = random.randint(0, len(dataset) - 1)
                        ds_out = dataset[idx]
                        self.assertIsInstance(ds_out, dict)
                        self.assertEqual(ds_out["target"].shape, (i, size, size))
                        if dataset.get_conditional_channel_size() == -1:
                            self.assertEqual(i, ds_out["condition"].shape[-3])
                        else:
                            self.assertEqual(dataset.get_conditional_channel_size(), ds_out["condition"].shape[-3])
                        self.assertEqual(ds_out["condition"].shape[-2:], (size, size))
            self.assertNotEqual(num_ds, 0, msg="download at least on dataset")

    def test_dataset_inpainting(self):
        from loader import load_dataset

        num_ds = 0
        for train in [True, False]:
            for dataset_name in load_dataset.dataset_names_img2img:
                with self.subTest(dataset=dataset_name):
                    root = load_dataset.get_potential_roots(dataset_name, offset=1)
                    if not os.path.isdir(root):
                        print(f"Skip dataset {dataset_name}, because it is not downloaded")
                        continue
                    num_ds += 1
                    size = 32
                    dataset, i = load_dataset.getDataset(
                        None, train, dataset=dataset_name, size=size, offset=1, inpainting="random_ege"
                    )
                    self.assertEqual(i, 3)

                    self.assertNotEqual(len(dataset), 0)
                    for _ in range(100):
                        idx = random.randint(0, len(dataset) - 1)
                        ds_out = dataset[idx]
                        self.assertIsInstance(ds_out, dict)
                        self.assertEqual(ds_out["target"].shape, (i, size, size))
                        if dataset.get_conditional_channel_size() == -1:
                            self.assertEqual(i, ds_out["condition"].shape[-3])
                        else:
                            self.assertEqual(dataset.get_conditional_channel_size(), ds_out["condition"].shape[-3])
                        self.assertEqual(ds_out["condition"].shape[-2:], (size, size))
                    break
            self.assertNotEqual(num_ds, 0, msg="download at least on dataset")

    def test_training(self):
        import diffusion as diffusion

        opt = diffusion.get_opt("config/Img2Img.conf")
        diffusion.main(opt, limit_train_batches=10)

    def test_training_unconditional(self):
        import diffusion as diffusion

        opt = diffusion.get_opt("config/unconditional.conf")
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
