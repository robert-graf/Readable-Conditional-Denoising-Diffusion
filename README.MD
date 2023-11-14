
[![DOI](https://zenodo.org/badge/605981537.svg)](https://zenodo.org/badge/latestdoi/605981537)
# Denoising diffusion-based MR to CT image translation enables whole spine vertebral segmentation in 2D and 3D without manual annotations.

This is the official 3D code for "Denoising diffusion-based MR to CT image translation enables whole spine vertebral segmentation in 2D and 3D without manual annotations."

![img](diffusion3d/images/example3D.jpg)
*This repository does not contain a segmentation algorithm.

## Content

This Repository contains image-2-image translation algorithms. In folder "img2img2D," you will find the 2D implementation of CUT, Pix2Pix, and Denoising Diffusion (DDPM and DDIM in image and noise mode). In the folder "diffusion3D", you will find the 3D variant of our Denoising Diffusion. We provide an introduction to how to run the 3D code and inference. We also pushed a pre-trained 3D Denoising Diffusion network (image mode) for sagittal T2w to CT (spine) translation and example data. 

## Citation

This is the official code for: 

Graf, R., Schmitt, J., Schlaeger, S. et al. Denoising diffusion-based MRI to CT image translation enables automated spinal segmentation. Eur Radiol Exp 7, 70 (2023).

[https://doi.org/10.1186/s41747-023-00385-2](https://doi.org/10.1186/s41747-023-00385-2)


## Funding 

The research for this article received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation program (101045128—iBack-epic—ERC2021-COG). 
