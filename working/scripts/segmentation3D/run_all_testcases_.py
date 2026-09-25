import sys
from pathlib import Path

sys.path.append(str(Path().parent.parent))
from scripts.reload_any import Reload_Any_Option
import scripts.reload_any as reload_any
import scripts.segmentation3D.test3D_help as test3D_help

# exp_names = ["diffusion3D_T2_CT", "dif_spinegan3D_T2_CT", "dif_spinegan3D_T2_CT_4_steps", "dif_spinegan3D_T2_CT_4_steps_noise"]
# log_dir_name = ["logs_diffusion", "logs_diffusion3D", "logs_diffusion3D", "logs_diffusion3D", "logs_diffusion3D"]

# exp_names = ["diffusion3D_T2_CT"]
exp_names = ["dif_spinegan3D_T2_CT_img", "diffusion3D_T2_CT", "CUT_diffusion3D_T2_CT"]
log_dir_name = ["logs_diffusion3D", "logs_diffusion", "logs_diffusion3D"]


parser = reload_any.get_option_reload()
parser.add_argument("-ds", default="All", choices=reload_any.translation_type(enum=test3D_help.DatasetEnum))
opt = reload_any.get_option(parser)
opt.timesteps = 25
opt.eta = 1
opt.guidance_w = 0
opt.ddim = True

for name, dir in zip(exp_names, log_dir_name):
    opt.exp_name = name
    opt.log_dir_name = dir
    if "CUT" in name:
        opt.cut = True
        opt.ddim = False
        opt.ddpm = False
    else:
        opt.ddim = True
        opt.cut = False
    # opt.exp_name = "diffusion3D_T2_CT"
    # opt.log_dir_name = "logs_diffusion"
    # opt.ds = "T2w"t
    # opt.ds = "Chl"
    # opt.ds = "Chl_our"
    test3D_help.run_dice_on_DS(opt)
