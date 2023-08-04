import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))
from scripts.segmentation.NAKO2Seg_vote import segment_with_network
import reload_any
from reload_any import Reload_Any_Option as ROpt
from threading import Thread


def segment_all(models_opt: list[ROpt], opt: ROpt, n_jobs=16):
    for opt_tmp in models_opt:
        print("#############################################################################")
        print(opt_tmp.get_result_folder_name())
        threads: list[Thread] = []
        try:
            t = segment_with_network(
                opt_tmp,
                match_string=opt.match_string,
                threading_post=True,
                snap=True,
                trans_type=opt.translationType,
            )
            threads.append(t)
        except Exception as e:
            print(e)
        for t in threads:
            t.join()


def get_all_models(root: str, T1: bool, opts={}, ablation_registration=False) -> list[ROpt]:
    if ablation_registration:
        if T1:
            return []
        models_opt = [
            ROpt("exp_syndiff", root, bs=20, syndiff=True, **opts),
            ROpt("paper_noreg_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("exp_syndiff_t1w_unpaird", root, bs=20, syndiff=True, **opts),
            ROpt("paper_noreg_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("paper_noreg_T2_pix2pix", root, bs=12, cut=True, **opts),
            ROpt("paper_1_reg_T2_pix2pix_sa-unet", root, bs=4, cut=True, **opts),
            ROpt("paper_noreg_T2_diffusion_img", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("paper_1_reg_T2_pix2pix", root, bs=12, cut=True, **opts),
            ROpt("paper_T2_pix2pix", root, bs=12, cut=True, **opts),
            ROpt(
                "paper_1_reg_T2_diffusion_img_malek", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts
            ),
            ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=10, eta=1, guidance_w=0, **opts),
            ROpt("exp_syndiff_t2w_0p", root, bs=20, syndiff=True, **opts),
            ROpt("exp_syndiff_t2w_1p", root, bs=20, syndiff=True, **opts),
            ROpt("exp_syndiff_t2w", root, bs=20, syndiff=True, **opts),
            ROpt("paper_1_reg_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
        ]
        return models_opt
    if T1:
        models_opt = [
            ROpt("paper_T1_cut", root, bs=8, cut=True, **opts),
            ROpt("paper_T1_cut_sa-unet", root, bs=8, cut=True, **opts),
            ROpt("paper_T1_pix2pix", root, bs=12, cut=True, **opts),
            ROpt("paper_T1_pix2pix_sa-unet", root, bs=4, cut=True, **opts),
            # ROpt("paper_T1_pcut_sa-unet", root, bs=20, cut=True, **opts),
            ROpt("exp_syndiff", root, bs=20, syndiff=True, **opts),
            ROpt("paper_T1_diffusion", root, bs=20, ddim=True, timesteps=10, eta=1, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion", root, bs=20, ddim=True, timesteps=20, eta=0, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=1, **opts),
            ROpt("paper_T1_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=2, **opts),
            ROpt("paper_T1_diffusion", root, bs=20, ddim=True, timesteps=50, eta=1, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion", root, bs=20, ddpm=True, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion_image", root, bs=20, ddim=True, timesteps=10, eta=1, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion_image", root, bs=20, ddim=True, timesteps=20, eta=0, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion_image", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion_image", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=1, **opts),
            ROpt("paper_T1_diffusion_image", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=2, **opts),
            # ROpt("paper_T1_diffusion_image", root, bs=20, ddim=True, timesteps=50, eta=0, guidance_w=0,**opts),
            ROpt("paper_T1_diffusion_image", root, bs=20, ddim=True, timesteps=50, eta=1, guidance_w=0, **opts),
            ROpt("paper_T1_diffusion_image", root, bs=20, ddpm=True, guidance_w=0, **opts),
        ]
    else:

        models_opt = [
            ROpt("paper_T2_cut", root, bs=20, cut=True, **opts),
            ROpt("paper_T2_cut_sa-unet", root, bs=20, cut=True, **opts),
            ROpt("paper_T2_pix2pix", root, bs=12, cut=True, **opts),
            ROpt("paper_T2_pix2pix_sa-unet", root, bs=8, cut=True, **opts),
            # ROpt("paper_T2_pcut_sa-unet", root, bs=20, cut=True, **opts),
            ROpt("exp_syndiff_t2w", root, bs=20, syndiff=True, **opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=10, eta=1, guidance_w=0, **opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=0, guidance_w=0, **opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=1, **opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=2, **opts),
            # ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=50, eta=0, guidance_w=0,**opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddim=True, timesteps=50, eta=1, guidance_w=0, **opts),
            ROpt("paper_T2_diffusion", root, bs=20, ddpm=True, guidance_w=0, **opts),
            ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=10, eta=1, guidance_w=0, **opts),
            ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=20, eta=0, guidance_w=0, **opts),
            ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=0, **opts),
            ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=1, **opts),
            ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=20, eta=1, guidance_w=2, **opts),
            # ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=50, eta=0, guidance_w=0,**opts),
            ROpt("paper_T2_ablation_image", root, bs=20, ddim=True, timesteps=50, eta=1, guidance_w=0, **opts),
            ROpt("paper_T2_ablation_image", root, bs=20, ddpm=True, guidance_w=0, **opts),
        ]
    # CORE
    core = True
    cut = False
    pix2pic = False
    ts = False
    eta = False
    ws = False
    ddpm = False
    if core:
        models_opt = [
            m
            for m in models_opt
            if m.exp_name.endswith("cut")
            or "syndiff" in m.exp_name
            or m.exp_name.endswith("pix2pix")
            or (m.timesteps == 20 and m.eta == 1 and m.guidance_w == 0)
        ]
    if cut:
        models_opt = [m for m in models_opt if "cut" in m.exp_name]
    if pix2pic:
        models_opt = [m for m in models_opt if "pix2pix" in m.exp_name]
    if ts:
        models_opt = [
            m for m in models_opt if m.ddim and m.eta == 1 and m.guidance_w == 0 and not m.exp_name.endswith("image")
        ]
    if eta:
        models_opt = [
            m
            for m in models_opt
            if m.ddim and m.timesteps == 20 and m.guidance_w == 0  # and m.exp_name.endswith("image")
        ]
    if ws:
        models_opt = [
            m for m in models_opt if m.ddim and m.timesteps == 20 and m.eta == 1  # and m.exp_name.endswith("image")
        ]
    if ddpm:
        models_opt = [models_opt[-1]]
    # models_opt = [m for m in models_opt if 1 == m.eta and m.exp_name.endswith("image")]
    return models_opt


if __name__ == "__main__":
    parser = reload_any.get_option_reload(True)
    opt = reload_any.get_option(parser)
    # opt.exp_name
    forbidden = ["cut", "ddim", "timesteps", "eta", "root", "bs", "exp_name", "config", "syndiff", "ddpm"]
    opts = {k: v for k, v in opt.__dict__.items() if k not in forbidden}

    segment_all(get_all_models(opt.root, opt.T1, opts), opt)

# CUDA_VISIBLE_DEVICES=0 python3 scripts/segmentation/NAKO2Seg_run_all.py --out /media/data/NAKO/MRT/test/individual2 --root /media/data/NAKO/MRT/rawdata/104/sub-104452-30/ --keep_resampled --resample ISO
# CUDA_VISIBLE_DEVICES=0 python3 scripts/segmentation/NAKO2Seg_run_all.py --out /media/data/NAKO/MRT/test/individual --root /media/data/NAKO/MRT/rawdata/104/sub-104852-30/ --keep_resampled --resample ISO
# CUDA_VISIBLE_DEVICES=0 python3 scripts/segmentation/NAKO2Seg_run_all.py --out /media/data/NAKO/MRT/test_T2w/ --root /media/data/NAKO/MRT/test_T2w/mri/ --match_string +/+/+acq-real_dixon.nii.gz --translationType TOP
# CUDA_VISIBLE_DEVICES=0 python3 scripts/segmentation/NAKO2Seg_run_all.py --out /media/data/NAKO/MRT/test_T1w/ --root /media/data/NAKO/MRT/test_T1w/mri/ --match_string +/+/+T1c.nii.gz --translationType TOP --T1
# CUDA_VISIBLE_DEVICES=0 python3 scripts/segmentation/NAKO2Seg_run_all.py --out /media/data/NAKO/MRT/test/new_docker --root /media/data/NAKO/MRT/rawdata/107/sub-10785+-30/ --keep_resampled
