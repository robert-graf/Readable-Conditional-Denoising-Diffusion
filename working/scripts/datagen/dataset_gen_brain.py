from genericpath import isdir
import random
import numpy as np
from pathlib import Path

from dataset_generation import load_nii, make_np_to_PNG, make_np_to_npz
from BIDS import NII

random.seed(8023)
image_types = ["arr"]

names = [
    "041d778862f07afb7c7b8c470aa442f1728b5166920748934090aeefb1689607",
    "08bf65c120e111e6e61ab95c7b0915b74e02694e82fddd2b0143cbe742c773c2",
    "09c607a6aa748852b41740be778fd99a51bbb8b19493d398cd25e2ac15779650",
    "10d784f43c6515cabf973028d83ab60bd1218488102f04a3b144f5e5b21ecfb7",
    "23d2a845f0d2fcc43d165ae9dba165382ff197a7b10e0cc5982c9306ba572301",
    "2dd40ab57edda37a6c94131fa5e123c545109d7949e028cbebe224be82ab19fa",
    "310483544c01d71c559bd51ae6f0e202783aabc37292cab092738448a70c03b4",
    "36539f01ec1d83872af0c904b5db4ab98129a3f56c86b75c6c83dc47b1a856d3",
    "52cf551ab5571339da253228712ce57d42b9f7b9d3da1dad1f2f4cfcda5d513a",
    "5418fa7895ce5366b4b0812eae3647eed7544539b9b688d421ce930846bc848e",
    "561f007f9c7e65a14457ccf2914595d8be0615a8ecaaa8798a69cdd3ef393c32",
    "5def39659ce9cac1ca5291520d65d61919abfcb4b89fbcf25040acf6d4309d70",
    "5fee191e0148f5575a950195d0edc87bfcfaa4de609c7961dbaa1425b065a0ed",
    "64b1703c499b4d465c40b709ce95b285aba3d484c070794f0d847e94b096a524",
    "75c6d1a0e465a28776efff6785432a1b224658d3e83e7cb7a141309d7aa8851a",
    "76f1f5c55f42b7eaa17aee60dd70141d33743f60e130d2ccdb275ea576c51a7e",
    "81a58d26ed53a4ac64c69abe25bd233d03f54f7edfe812ec7ed4df3f16d9bd3a",
    "89a792c83d56ae7f6fc711b23e841dc53e11a6911423dac90c47f494ddc9599e",
    "95d8ee0ef2a13f13d2a6c0f7464ea8c2965875974ba363b13d34118bf4fa70e7",
    "9c9e84e817fa648e9e39648915c1a2d5d70d6b5238b116558f753a7d999932fc",
    "a0775625d6459164dd6e9236a0f27605c6ca02db67be9e406a95bf7d6e7fe0cf",
    "a7e1cdd9e310e802cb4bacdc1b17e3b5405e9a7e804247602acc7bbfe2ef1702",
    "a877bf15a3b83d352d902caf9aa40a726bb41f626e39dbbdbc42262aa5086de8",
    "ae1bb6a9a75162c9e0f9f3867de98f970765a1e609c44f3657ff56203b9465b3",
    "b4a377f30a60381361e99caf95c866c8c8181c85912606e3e4f4bfb2e8ddd638",
    "c76ca92b37ba7b429c07011acfd2cbd1507ee748872f7c351adb9610815872c5",
    "ce6e30c3987144f75b90ffa32f0571e6dc5c5d7e154800e830c53acc655d32fe",
    "d73a3d518515032910622b7c9dbe1d7540965675525d550920221980aea23672",
    "e3499edcc1b171bb19b2ee5a7f1bf733cf9376879ae51001eb6608801aca262f",
    "e8332b3b0d3b2ae7a2fe203e790c6519eaa9cbfdc7968d395525e6bd541a4696",
    "f9c29027297083e3813505df6373fec1f2a154a6760c5f5031be1ac1d8510d36",
    "fe814fd0daede3baa4b72155b5eec4b37f403c236814a1d8f6ef6c74e832bbf2",
    "m_002348",
    "m_003442",
    "m_016399",
    "m_025193",
    "m_043194",
    "m_055235",
    "m_066632",
    "m_072493",
    "m_073174",
    "m_075603",
    "m_084008",
    "m_084036",
    "m_095271",
    "m_113859",
    "m_122476",
    "m_129380",
    "m_139339",
    "m_160071",
    "m_173884",
    "m_186170",
    "m_188028",
    "m_193903",
    "m_195509",
    "m_199612",
    "m_203013",
    "m_211445",
    "m_236853",
    "m_275415",
    "m_317996",
    "m_321757",
    "m_329022",
    "m_330355",
    "m_335467",
    "m_335984",
    "m_338260",
    "m_348904",
    "m_353646",
    "m_362608",
    "m_372924",
    "m_374823",
    "m_375999",
    "m_381209",
    "m_385242",
    "m_391165",
    "m_425924",
    "m_451978",
    "m_460902",
    "m_465922",
    "m_469206",
    "m_472036",
    "m_479542",
    "m_514993",
    "m_543028",
    "m_600098",
    "m_612067",
    "m_617099",
    "m_629538",
    "m_632272",
    "m_664510",
    "m_679803",
    "m_697628",
    "m_705667",
    "m_710706",
    "m_717375",
    "m_729767",
    "m_737152",
    "m_752194",
    "m_778510",
    "m_788898",
    "m_790858",
    "m_800113",
    "m_804068",
    "m_815005",
    "m_815123",
    "m_815232",
    "m_816146",
    "m_829230",
    "m_856319",
    "m_876032",
    "m_886633",
    "m_893044",
    "m_903168",
    "m_904552",
    "m_904705",
    "m_909804",
    "m_913745",
    "m_920933",
    "m_938793",
    "m_942861",
    "m_947499",
    "m_966414",
    "m_986156",
    "m_993878",
    "m_999634",
]

mapping = {k: random.randint(1, 10) for k in names}


def setup(file: Path, orientation):
    nii = NII.load(file, seg=False).reorient(orientation)
    sg = nii.get_array()
    sg[sg >= 20] = 1
    arr_dic: dict[str, np.ndarray] = {"arr": nii.get_array(), "sg": sg}
    normalize = {"arr": lambda x, volumes: x / max(0.00001, np.max(volumes)), "sg": lambda x, y: x}
    interpolation_lvl = {"arr": 3, "sg": 1}
    return arr_dic, interpolation_lvl, normalize, "sg"


out = "/DATA/NAS/ongoing_projects/robert/train_dataset/brain"

if __name__ == "__main__":
    for search_path in ["/DATA/NAS/datasets_processed/brain/NeRF_MS_data/"]:
        try:
            for file in Path(search_path).rglob("**/*.nii.gz"):
                if "LR" in file.name:
                    continue
                if "mask" in file.name:
                    continue

                name = file.name
                sub = name.split("_ses")[0][4:]
                print(name)
                if mapping[sub] == 1:
                    out_path = f"{out}/val/"
                elif mapping[sub] == 7:
                    out_path = f"{out}/test/"
                else:
                    out_path = f"{out}/train/"

                for orientation in [
                    ("R", "I", "P"),  # RIP
                    ("I", "P", "R"),  # IRP
                    ("P", "I", "R"),  # PIR
                ]:
                    try:
                        arr_dic, interpolation_lvl, normalize, filter = setup(file, orientation)

                        s = "_" + orientation[0] + orientation[1] + orientation[2]
                        make_np_to_npz(
                            out_path,
                            name + s,
                            arr_dic=arr_dic,
                            interpolation_lvl=interpolation_lvl,
                            normalize=normalize,
                            filter=filter,
                            deform=False,
                        )
                    except Exception as ex:
                        print(ex)

        except Exception as e:
            raise e
