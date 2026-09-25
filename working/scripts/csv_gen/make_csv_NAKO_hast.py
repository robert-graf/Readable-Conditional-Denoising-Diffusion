from pathlib import Path
from BIDS import BIDS_Global_info, BIDS_FILE, NII
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
# print(sys.path)
from dataloader import fastnumpyio

dataset_path = "/media/data/NAKO/MRT/dataset-nako/"
dataset = BIDS_Global_info([dataset_path], parents=["rawdata"])

import random
from BIDS.logger.log_file import Reflection_Logger

logger = Reflection_Logger()
random.seed(42)
import numpy as np
from BIDS import to_nii
from joblib import Parallel, delayed


def rest_path(x: BIDS_FILE | Path | None):
    if x is None:
        return None
    if isinstance(x, BIDS_FILE):
        x = x.file["nii.gz"]
    par = x.parent.name
    return par + "/" + x.name


out_folder = Path("/media/data/robert/datasets/dataset-hast2")
out_folder.mkdir(exist_ok=True)
from tqdm import tqdm


def crop_slice2D(msk: np.ndarray, min_value=0):
    shp = msk.shape
    assert len(shp) == 2, shp
    cor_msk = np.where(msk > min_value)
    c_min = [cor_msk[0].min(), cor_msk[1].min()]
    c_max = [cor_msk[0].max(), cor_msk[1].max()]
    x0 = c_min[0] if c_min[0] > 0 else 0
    y0 = c_min[1] if c_min[1] > 0 else 0
    x1 = c_max[0] if c_max[0] < shp[0] else shp[0]
    y1 = c_max[1] if c_max[1] < shp[1] else shp[1]
    ex_slice = tuple([slice(x0, x1), slice(y0, y1)])
    print(shp, ex_slice)
    return ex_slice


tasks = []
for name, subj in tqdm(dataset.enumerate_subjects(), total=len(dataset)):
    q = subj.new_query()
    q.flatten()
    q.filter_format(["T2w"])
    q2 = subj.new_query()
    q2.flatten()
    q2.filter_format(["T2haste"])

    split = random.random() * 25
    if split < 0.8:
        phase = "train"
    elif split < 0.85:
        phase = "val"
    elif split < 1.0:
        phase = "test"
    else:
        continue
    try:
        # def run(q):
        t2_haste = list(q2.loop_list())[0]
        t2w = list(q.loop_list())
        t2w_bids = t2w[random.randint(0, len(t2w) - 1)]

        t2w = to_nii(t2w_bids).reorient_(("R", "I", "P"))
        t2_haste = to_nii(t2_haste)
        t2_haste.reorient_(("R", "I", "P")).rescale_(t2w.zoom)
        t2w.resample_from_to_(t2_haste, verbose=False)
        bb = t2w.compute_crop_slice()
        bb = t2w.compute_crop_slice(other_crop=bb)
        t2w.apply_crop_slice_(bb)
        t2_haste.apply_crop_slice_(bb)
        # t2_haste.save(out_folder / f"{name}_T2hast.nii.gz")
        # t2w.save(out_folder / f"{name}_T2w.nii.gz")
        Path(out_folder, phase).mkdir(exist_ok=True)

        def make_new(t2w: NII, hast: NII, name):
            arrs = [t2w.normalize(quantile=0.95).get_array(), hast.normalize(quantile=0.95).get_array()]

            for i in range(t2w.shape[0]):
                tmp = arrs[0][i]

                if tmp.sum() <= 500:
                    continue
                crop = crop_slice2D(tmp)
                if crop[0].stop - crop[0].start < 100:
                    continue
                if crop[1].stop - crop[1].start < 100:
                    continue
                print(crop)
                arr = np.stack([arr[i][crop] for arr in arrs])
                assert len(arr.shape) == 3, (arr.shape, i)

                fastnumpyio.save(Path(out_folder, phase, f"{name}_{i:03}.numpy"), arr)

        # make_new(t2w, t2_haste, f"{name}_{t2w_bids.get('chunk')}")
        tasks.append(delayed(make_new)(t2w, t2_haste, f"{name}_{t2w_bids.get('chunk')}"))

        if len(tasks) >= 2:
            Parallel(n_jobs=24)(tasks)
            tasks = []
    except Exception as e:
        print(e)
Parallel(n_jobs=24)(tasks)
