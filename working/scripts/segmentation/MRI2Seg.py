import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[2]))
sys.path.append(str(file.parents[1]))

from scripts.segmentation.MRISeg_helper import segment_with_network
import reload_any

parser = reload_any.get_option_reload()
opt = reload_any.get_option(parser)
# "*-1_t2.nii*"
segment_with_network(opt, snap=True, match_string=opt.match_string)

# CUDA_VISIBLE_DEVICES=1 python3 scripts/NAKO2Seg.py --ddim --timesteps 20 --eta 0 -en bailiang_256 --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/110/sub-110815-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --cut --timesteps 20 --eta 0  -en paper_T2_cut             --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --cut --timesteps 20 --eta 0  -en paper_T2_cut_sa-unet     --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --cut --timesteps 20 --eta 0  -en paper_T2_pcut_sa-unet    --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --cut --timesteps 20 --eta 0 -en paper_T2_pix2pix_sa-unet  --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --cut --timesteps 20 --eta 0  -en paper_T2_pix2pix         --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 20 --eta 0 -en paper_T2_diffusion       --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 20 --eta 1 -en paper_T2_diffusion       --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 50 --eta 0 -en paper_T2_diffusion       --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 50 --eta 1 -en paper_T2_diffusion       --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddpm --timesteps 50 --eta 1 -en paper_T2_diffusion       --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/106/sub-106852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 20 --eta 0 -en bailiang_256 --out /media/data/NAKO/MRT/test/ --root /media/data/NAKO/MRT/rawdata/110/sub-110815-30/
#
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --syndiff -en   exp_syndiff_t2w   --out /media/data/NAKO/MRT/test/paper --root /media/data/NAKO/MRT/rawdata/104/sub-104852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --cut --timesteps 20 --eta 0  -en paper_T2_pcut_sa-unet      --out /media/data/NAKO/MRT/test/paper --root /media/data/NAKO/MRT/rawdata/104/sub-104852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --cut --timesteps 20 --eta 0  -en paper_T2_pix2pix_sa-unet     --out /media/data/NAKO/MRT/test/paper --root /media/data/NAKO/MRT/rawdata/104/sub-104852-30/


# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 20 --eta 0 --guidance_w 0  -en paper_T2_diffusion  --out /media/data/NAKO/MRT/test/paper --root /media/data/NAKO/MRT/rawdata/104/sub-104852-30/


# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 20 --eta 1 --guidance_w 0  -en paper_T2_ablation_image  --out /media/data/NAKO/MRT/test/paper --root /media/data/NAKO/MRT/rawdata/104/sub-104852-30/
# python3 scripts/segmentation/NAKO2Seg.py --keep_cdt --ddim --timesteps 20 --eta 1 --guidance_w 0  -en paper_T2_diffusion  --out /media/data/NAKO/MRT/test/paper --root /media/data/NAKO/MRT/rawdata/104/sub-104852-30/


# python3 scripts/segmentation/NAKO2Seg.py --ddim --timesteps 20 --eta 1 --guidance_w 0  -en paper_T2_ablation_image  --out /media/data/robert/datasets/spinegan_T2w/transl/ --root /media/data/robert/datasets/spinegan_T2w/registration/spinegan0038/target-803_from-11/ --match_string +l_dixon.nii.gz --translationType TOP
# python3 scripts/segmentation/NAKO2Seg.py --cut --timesteps 20 --eta 0  -en paper_T2_pix2pix_sa-unet   --out /media/data/robert/datasets/spinegan_T2w/transl/ --root /media/data/robert/datasets/spinegan_T2w/registration/spinegan0038/target-803_from-11/ --match_string +l_dixon.nii.gz --translationType TOP
#
# python3 scripts/segmentation/NAKO2Seg.py --syndiff --timesteps 20 --eta 0  -en exp_syndiff_t2w   --out /media/data/robert/datasets/spinegan_T2w/translation/ --root /media/data/robert/datasets/spinegan_T2w/sourcedata/ --match_string +/+/+_dixon.nii.gz --translationType V_STITCHING --filter_dixon
