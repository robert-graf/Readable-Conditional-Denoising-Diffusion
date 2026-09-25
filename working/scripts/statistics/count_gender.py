from __future__ import annotations
from dataclasses import dataclass, field
from email.policy import default
from pathlib import Path
import sys

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))

from BIDS.bids_files import BIDS_FILE, BIDS_Global_info, Searchquery, Subject_Container


@dataclass()
class image_class:
    datasets: str = "/media/data/robert/datasets/dataset_spinegan"
    parents: list[str] = field(default_factory=lambda: ["rawdata", "rawdata_ct", "rawdata_dixon"])


keys = ["PatientBirthDate", "PatientSex", "PatientAge", "PatientWeight"]

sequ_key = {}


@dataclass()
class sub_counter:
    name: str
    count: dict = field(default_factory=lambda: {"ct": 0, "dixon": 0})
    meta: dict[str, dict[str, list[str]]] = field(default_factory=lambda: {"ct": {}})  # "dixon": dict.fromkeys(keys, [].copy()


i = image_class()

glob_bids = BIDS_Global_info([i.datasets], i.parents)

subjects_dict: dict[str, sub_counter] = {}


for name, sub in glob_bids.enumerate_subjects():
    # if name != "spinegan0012":
    #    continue
    sub_count = sub_counter(name)
    for sequ_key, value in sub.sequences.items():
        print(sequ_key)
        for v in value:
            if v.format == "ct":
                sub_count.count["ct"] += 1
                dic = v.open_json()
                for key in keys:
                    x = dic.get(key, None)
                    if x is None:
                        continue
                    if key not in sub_count.meta["ct"]:
                        sub_count.meta["ct"][key] = []
                    if x in sub_count.meta["ct"][key]:
                        continue
                    sub_count.meta["ct"][key].append(x)

                    # "PatientBirthDate": "19550101",
                    # "PatientSex": "F",
                    # "PatientAge": "065Y",
                    # "PatientWeight": 97.0,
            print(v)

    subjects_dict[name] = sub_count

print(subjects_dict["spinegan0012"])
# Count M
male = 0
female = 0
male_age = []
female_age = []
no_age = 0


def get_age(v: sub_counter):
    try:
        age = v.meta["ct"]["PatientAge"]
        assert len(age) == 1
        age = float(age[0].replace("Y", ""))
        return age
    except:
        return None


for k, v in subjects_dict.items():
    if v.count["ct"] == 0:
        continue
    print(k, v)
    sex = v.meta["ct"]["PatientSex"]
    assert len(sex) == 1
    if sex[0] == "M":
        male += 1
        age = get_age(v)
        if age is None:
            no_age += 1
        else:
            male_age.append(age)

    elif sex[0] == "F":
        female += 1
        age = get_age(v)
        if age is None:
            no_age += 1
        else:
            female_age.append(age)
    else:
        assert False, sex
from statistics import mean, stdev

print(
    f"\tmale\t {male} {mean(male_age): 3.1f} ± {stdev(male_age): 3.1f} \n",
    f"\tfemale\t {female} {mean(female_age): 3.1f} ± {stdev(female_age): 3.1f} \n",
    no_age,
)
