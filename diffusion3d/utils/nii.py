from typing import Tuple
from math import ceil, floor
from typing import TypeVar, Union
from pathlib import Path
import warnings
import numpy as np
import nibabel as nib

import nibabel.processing as nip
import nibabel.orientations as nio
import traceback
from typing_extensions import Self
from nibabel import Nifti1Image, Nifti1Header  # type: ignore
from typing import Type
from utils.nii_math import NII_Math

from typing import Tuple, Dict, Literal, Callable

# R: Right, L: Left; S: Superio (up), I: Inverior (down); A: Anterior (front), P: Posterior (back)
Directions = Literal["R", "L", "S", "I", "A", "P"]
Ax_Codes = tuple[Directions, Directions, Directions]
LABEL_MAX = 256
Zooms = Tuple[float, float, float]

Centroid_Dict = Dict[int, Tuple[float, float, float]]
Coordinate = Tuple[float, float, float]
POI_Dict = Dict[int, Dict[int, Coordinate]]
import numpy as np

Rotation = np.ndarray
Label_Map = dict[int | str, int | str] | dict[str, str] | dict[int, int]

_formatwarning = warnings.formatwarning


def formatwarning_tb(*args, **kwargs):
    s = "####################################\n"
    s += _formatwarning(*args, **kwargs)
    tb = traceback.format_stack()[:-3]
    s += "".join(tb[:-1])
    s += "####################################\n"
    return s


warnings.formatwarning = formatwarning_tb

N = TypeVar("N", bound="NII")
Image_Reference = Union[Nifti1Image, Path, str, N]
Interpolateable_Image_Reference = Union[
    Tuple[Nifti1Image, bool],
    Tuple[Path, bool],
    Tuple[str, bool],
    N,
]
Proxy = Tuple[Tuple[int, int, int], np.ndarray]
suppress_dtype_change_printout_in_set_array = False
# fmt: off
class NII(NII_Math):
    def __init__(self, nii: Nifti1Image, seg=False,c_val=None) -> None:
        self.nii:Nifti1Image = nii
        self.seg:bool = seg
        self.c_val:float|None=c_val # default c_vale if seg is None
        self.__min = None 
    @classmethod
    def load(cls, path: Image_Reference, seg, c_val=None):
        nii= to_nii(path,seg)
        nii.c_val = c_val
        return nii
        #return NII(nib.load(path), seg, c_val) #type: ignore
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.nii.shape
    @property
    def dtype(self)->Type:
        return self.nii.dataobj.dtype #type: ignore
    @property
    def header(self) -> Nifti1Header:
        return self.nii.header  # type: ignore

    @property
    def affine(self) -> np.ndarray:
        return self.nii.affine

    @property
    def orientation(self) -> Ax_Codes:
        ort = nio.io_orientation(self.affine)
        return nio.ornt2axcodes(ort)

    @property
    def zoom(self) -> tuple[float, float, float]:
        z = self.header.get_zooms()
        assert len(z) == 3
        return z
    @property
    def origin(self) -> tuple[float, float, float]:
        z = tuple(np.round(self.nii.affine[:3,3],7))
        assert len(z) == 3
        return z
    @property
    def rotation(self)->np.ndarray:
        rotation_zoom = self.affine[:3, :3]
        zoom = np.array(self.zoom)
        rotation = rotation_zoom / zoom
        return rotation
    
    
    @orientation.setter
    def orientation(self, value: Ax_Codes):
        self.reorient_(value, verbose=False)

    @property
    def orientation_ornt(self):
        return nio.io_orientation(self.affine)
    
    def get_c_val(self,default=None):
        if self.seg:
            return 0
        if self.c_val is not None:
            return self.c_val
        if default is not None:
            return default
        if self.__min is None:
            self.__min = self.min()
        return self.__min
        
    def get_seg_array(self) -> np.ndarray:
        if not self.seg:
            warnings.warn(
                "requested a segmentation array, but NII is not set as a segmentation", UserWarning, stacklevel=5
            )
        return np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).astype(np.uint16).copy() #type: ignore
    def _extract_affine(self):
        return {"zoom":self.zoom,"origin": self.origin, "shape": self.shape, "rotation": self.rotation}
    def get_array(self) -> np.ndarray:
        if self.seg:
            return self.get_seg_array()
        return np.asanyarray(self.nii.dataobj, dtype=self.nii.dataobj.dtype).copy() #type: ignore
    def set_array(self,arr:np.ndarray, inplace=False,verbose=True)-> Self:
        """Creates a NII where the array is replaces with the input array. 
        
        Note: This function works "Out-of-place" by default, like all other methods.

        Args:
            arr (np.ndarray): _description_
            inplace (bool, optional): _description_. Defaults to False.

        Returns:
            self
        """
        if arr.dtype == bool:
            arr = arr.astype(np.uint8)
            
        if self.nii.dataobj.dtype == arr.dtype: #type: ignore
            nii = Nifti1Image(arr,self.affine,self.header)
        else:
            if not suppress_dtype_change_printout_in_set_array:
                print(f"'set_array' with different dtype: from {self.nii.dataobj.dtype} to {arr.dtype}") if verbose else None #type: ignore
            nii = Nifti1Image(self.get_array(),self.affine,self.header)
            nii.set_data_dtype(arr.dtype)
            nii = Nifti1Image(arr,nii.affine,nii.header)
        if all(a is None for a in self.header.get_slope_inter()):
            nii.header.set_slope_inter(1,self.get_c_val()) # type: ignore
        
        if inplace:
            self.nii = nii
            return self
        else:
            return NII(nii,self.seg)
    
    def set_array_(self,arr:np.ndarray,verbose=True):
        return self.set_array(arr,inplace=True,verbose=verbose)
    def set_dtype_(self,dtype:Type|Literal['smallest_int'] = np.float32):
        if dtype == "smallest_int":
            arr = self.get_array()
            if arr.max()<128:
                dtype = np.int8
            elif arr.max()<32768:
                dtype = np.int16
            else:
                dtype = np.int32
        
        self.nii.set_data_dtype(dtype)
        if self.nii.get_data_dtype() != self.nii.dataobj.dtype: #type: ignore
            self.nii = Nifti1Image(self.get_array().astype(dtype),self.affine,self.header)
            
        return self
    def global_to_local(self, x: tuple[float, float, float] | list[float]):
        a =  self.affine[:3, :3].T.dot(np.array(x) - self.affine[:3, 3])  # type:ignore
        return tuple(round(float(v), 7) for v in a)

    def local_to_global(self, x: tuple[float, float, float] | list[float]):
        a= self.affine[:3, :3].dot(np.array(x)) + self.affine[:3, 3]  # type:ignore
        return tuple(round(float(v), 7) for v in a)


    def reorient(self:Self, axcodes_to: Ax_Codes = ("P", "I", "R"), verbose=False, inplace=False)-> Self:
        """
        Reorients the input Nifti image to the desired orientation, specified by the axis codes.

        Args:
            axcodes_to (tuple): A tuple of three strings representing the desired axis codes. Default value is ("P", "I", "R").
            verbose (bool): If True, prints a message indicating the orientation change. Default value is False.
            inplace (bool): If True, modifies the input image in place. Default value is False.

        Returns:
            If inplace is True, returns None. Otherwise, returns a new instance of the NII class representing the reoriented image.

        Note:
        The nibabel axes codes describe the direction, not the origin, of axes. The direction "PIR+" corresponds to the origin "ASL".
        """
        # Note: nibabel axes codes describe the direction not origin of axes
        # direction PIR+ = origin ASL

        aff = self.affine
        ornt_fr = self.orientation_ornt
        arr = self.get_array()
        ornt_to = nio.axcodes2ornt(axcodes_to)
        ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
        if (ornt_fr == ornt_to).all():
            print("Image is already rotated to", axcodes_to) if verbose else None
            return self
        arr = nio.apply_orientation(arr, ornt_trans)
        aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
        new_aff = np.matmul(aff, aff_trans)
        ### Reset origin ###
        flip = ornt_trans[:, 1]
        change = ((-flip) + 1) / 2  # 1 if flip else 0
        change = tuple(a * s for a, s in zip(change, self.shape))
        new_aff[:3, 3] = nib.affines.apply_affine(aff,change) # type: ignore
        ######
        new_img = Nifti1Image(arr, new_aff,self.header)
        if all(a is None for a in self.header.get_slope_inter()):
            new_img.header.set_slope_inter(1,self.get_c_val()) # type: ignore
        print("Image reoriented from", nio.ornt2axcodes(ornt_fr), "to", axcodes_to) if verbose else None
        if inplace:
            self.nii = new_img
            return self

        return NII(new_img, self.seg) # type: ignore
    def reorient_(self:Self, axcodes_to: Ax_Codes|None = ("P", "I", "R"), verbose=False) -> Self:
        if axcodes_to is None:
            return self
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose,inplace=True)
    
    def compute_crop_slice(self,minimum=0, dist=0, other_crop:tuple[slice,...]|None=None, minimum_size:tuple[slice,...]|int|tuple[int,...]|None=None):
        """
        Computes the minimum slice that removes unused space from the image and returns the corresponding slice tuple along with the origin shift required for centroids.

        Args:
            minimum (int): The minimum value of the array (0 for MRI, -1024 for CT). Default value is 0.
            dist (int): The amount of padding to be added to the cropped image. Default value is 0.
            other_crop (tuple[slice,...], optional): A tuple of slice objects representing the slice of an other image to be combined with the current slice. Default value is None.

        Returns:
            ex_slice: A tuple of slice objects that need to be applied to crop the image.
            origin_shift: A tuple of integers representing the shift required to obtain the centroids of the cropped image.

        Note:
            - The computed slice removes the unused space from the image based on the minimum value.
            - The padding is added to the computed slice.
            - If the computed slice reduces the array size to zero, a ValueError is raised.
            - If other_crop is not None, the computed slice is combined with the slice of another image to obtain a common region of interest.
            - Only None slice is supported for combining slices.
        """  
        shp = self.shape
        zms = self.header.get_zooms()
        d = np.around(dist / np.asarray(zms)).astype(int)
        array = self.get_array() #+ minimum
        msk_bin = np.zeros(array.shape,dtype=bool)
        #bool_arr[array<minimum] = 0
        msk_bin[array>minimum] = 1
        #msk_bin = np.asanyarray(bool_arr, dtype=bool)
        msk_bin[np.isnan(msk_bin)] = 0
        cor_msk = np.where(msk_bin > 0)
        if cor_msk[0].shape[0] == 0:
            raise ValueError('Array would be reduced to zero size')
        c_min = [cor_msk[0].min(), cor_msk[1].min(), cor_msk[2].min()]
        c_max = [cor_msk[0].max(), cor_msk[1].max(), cor_msk[2].max()]
        x0 = c_min[0] - d[0] if (c_min[0] - d[0]) > 0 else 0
        y0 = c_min[1] - d[1] if (c_min[1] - d[1]) > 0 else 0
        z0 = c_min[2] - d[2] if (c_min[2] - d[2]) > 0 else 0
        x1 = c_max[0] + d[0] if (c_max[0] + d[0]) < shp[0] else shp[0]
        y1 = c_max[1] + d[1] if (c_max[1] + d[1]) < shp[1] else shp[1]
        z1 = c_max[2] + d[2] if (c_max[2] + d[2]) < shp[2] else shp[2]
        ex_slice = [slice(x0, x1+1), slice(y0, y1+1), slice(z0, z1+1)]
        
        if other_crop is not None:
            assert all([(a.step == None) for a in other_crop]), 'Only None slice is supported for combining x'
            ex_slice = [slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(ex_slice, other_crop)]
        
        if minimum_size is not None:
            if isinstance(minimum_size,int):
                minimum_size = (minimum_size,minimum_size,minimum_size)
            for i, min_w in enumerate(minimum_size):
                if isinstance(min_w,slice):
                    min_w = min_w.stop - min_w.start
                curr_w =  ex_slice[i].stop - ex_slice[i].start
                dif = min_w - curr_w
                if min_w > 0:
                    new_start = ex_slice[i].start - floor(dif/2)
                    new_goal = ex_slice[i].stop + ceil(dif/2)
                    if new_goal > self.shape[i]:
                        new_start -= new_goal - self.shape[i]
                        new_goal = self.shape[i]
                    if new_start < 0:#
                        new_goal -= new_start
                        new_start = 0
                    ex_slice[i] = slice(new_start,new_goal)
                        
                        
        #origin_shift = tuple([int(ex_slice[i].start) for i in range(len(ex_slice))])
        return tuple(ex_slice)#, origin_shift

    def apply_crop_slice(self,ex_slice:tuple[slice,slice,slice] , inplace=False):
        """
        The apply_crop_slice function applies a given slice to reduce the Nifti image volume. If a list of slices is provided, it computes the minimum volume of all slices and applies it.

        Args:
            ex_slice (tuple[slice,slice,slice] | list[tuple[slice,slice,slice]]): A tuple or a list of tuples, where each tuple represents a slice for each axis (x, y, z).
            inplace (bool, optional): If True, it applies the slice to the original image and returns it. If False, it returns a new NII object with the sliced image.
        Returns:
            NII: A new NII object containing the sliced image if inplace=False. Otherwise, it returns the original NII object after applying the slice.
        """        ''''''
        nii = self.nii.slicer[ex_slice]
        if inplace:
            self.nii = nii
            return self
        return NII(nii,self.seg)
    
    def apply_crop_slice_(self,ex_slice:tuple[slice,slice,slice]):
        return self.apply_crop_slice(ex_slice=ex_slice,inplace=True)

    def rescale_and_reorient(self, axcodes_to=None, voxel_spacing=(-1, -1, -1), verbose=True, inplace=False,c_val:float|None=None,mode='constant'):
        
        ## Resample and rotate and Save Tempfiles
        if axcodes_to is None:
            curr = self
            ornt_img = self.orientation
            axcodes_to = nio.ornt2axcodes(ornt_img)
        else:
            curr = self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
        return curr.rescale(voxel_spacing=voxel_spacing, verbose=verbose, inplace=inplace,c_val=c_val,mode=mode)

    def rescale_and_reorient_(self,axcodes_to=None, voxel_spacing=(-1, -1, -1),c_val:float|None=None,mode='constant', verbose=True):
        return self.rescale_and_reorient(axcodes_to=axcodes_to,voxel_spacing=voxel_spacing,c_val=c_val,mode=mode,verbose=verbose,inplace=True)
    
    def reorient_same_as(self, img_as: Nifti1Image | Self, verbose=False, inplace=False) -> Self:
        axcodes_to = nio.ornt2axcodes(nio.io_orientation(img_as.affine))
        return self.reorient(axcodes_to=axcodes_to, verbose=verbose, inplace=inplace)
    def reorient_same_as_(self, img_as: Nifti1Image | Self, verbose=False) -> Self:
        return self.reorient_same_as(img_as=img_as,verbose=verbose,inplace=True)
    def rescale(self, voxel_spacing=(1, 1, 1), c_val:float|None=None, verbose=False, inplace=False,mode='constant'):
        """
        Rescales the NIfTI image to a new voxel spacing.

        Args:
            voxel_spacing (tuple[float, float, float]): The desired voxel spacing in millimeters (x, y, z). -1 is keep the voxel spacing.
                Defaults to (1, 1, 1). 
            c_val (float | None, optional): The padding value. Defaults to None, meaning that the padding value will be
                inferred from the image data.
            verbose (bool, optional): Whether to print a message indicating that the image has been resampled. Defaults to
                False.
            inplace (bool, optional): Whether to modify the current object or return a new one. Defaults to False.
            mode (str, optional): One of the supported modes by scipy.ndimage.interpolation (e.g., "constant", "nearest",
                "reflect", "wrap"). See the documentation for more details. Defaults to "constant".

        Returns:
            NII: A new NII object with the resampled image data.
        """
        if voxel_spacing == (-1,-1,-1) or voxel_spacing == self.zoom:
            return self.copy() if inplace else self
        
        c_val = self.get_c_val(c_val)
        # resample to new voxel spacing based on the current x-y-z-orientation
        aff = self.affine
        shp = self.shape
        zms = self.zoom
        order = 0 if self.seg else 3
        voxel_spacing = tuple([v if v != -1 else z for v,z in zip(voxel_spacing,zms)])
        if voxel_spacing == self.zoom:
            return self.copy() if inplace else self
        
        # Calculate new shape
        new_shp = tuple(np.rint([shp[i] * zms[i] / voxel_spacing[i] for i in range(len(voxel_spacing))]).astype(int))
        new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)  # type: ignore
        new_aff[:3, 3] = nib.affines.apply_affine(aff, [0, 0, 0])# type: ignore
        new_img = nip.resample_from_to(self.nii, (new_shp, new_aff), order=order, cval=c_val,mode=mode)
        print(f"Image resampled from {zms} to voxel size {voxel_spacing}") if verbose else None
        if inplace:
            self.nii = new_img
            return self
        return NII(new_img, self.seg,self.c_val)
    
    def rescale_(self, voxel_spacing=(1, 1, 1), c_val:float|None=None, verbose=False,mode='constant'):
        return self.rescale( voxel_spacing=voxel_spacing, c_val=c_val, verbose=verbose,mode=mode, inplace=True)
    
    def resample_from_to(self, to_vox_map:Image_Reference|Proxy, mode='constant', c_val=None, inplace = False,verbose=True):
        """self will be resampled in coordinate of given other image. Adheres to global space not to local pixel space

        Args:
            to_vox_map (Image_Reference|Proxy): If object, has attributes shape giving input voxel shape, and affine giving mapping of input voxels to output space. If length 2 sequence, elements are (shape, affine) with same meaning as above. The affine is a (4, 4) array-like.\n
            mode (str, optional): Points outside the boundaries of the input are filled according to the given mode ('constant', 'nearest', 'reflect' or 'wrap').Defaults to 'constant'.\n
            cval (float, optional): Value used for points outside the boundaries of the input if mode='constant'. Defaults to 0.0.\n
            inplace (bool, optional): Defaults to False.

        Returns:
            NII: 
        """        ''''''
        c_val = self.get_c_val(c_val)
            
        map = to_nii_optional(to_vox_map,seg=self.seg,default=to_vox_map)
        print(f"resample_from_to: {self} to {map}") if verbose else None
        
        nii = nip.resample_from_to(self.nii, map, order=0 if self.seg else 3, mode=mode, cval=c_val)
        if inplace:
            self.nii = nii
            return self
        else:
            return NII(nii,self.seg,self.c_val)
    def resample_from_to_(self, to_vox_map, mode='constant', c_val:float|None=None,verbose=True):
        return self.resample_from_to(to_vox_map,mode=mode,c_val=c_val,inplace=True,verbose=verbose)
    
    def n4_bias_field_correction(
        self,
        threshold = 60,
        mask=None,
        shrink_factor=4,
        convergence={"iters": [50, 50, 50, 50], "tol": 1e-07},
        spline_param=200,
        verbose=False,
        weight_mask=None,
        crop=False,
        inplace=False
    ):  
        assert self.seg == False
        # install antspyx not ants!
        import ants.utils.bias_correction as bc # install antspyx not ants!
        from ants.utils.convert_nibabel import from_nibabel
        from scipy.ndimage import generate_binary_structure, binary_dilation
        dtype = self.dtype
        input_ants = from_nibabel(self.nii)
        if threshold != 0:
            mask = self.get_array()
            mask[mask < threshold] = 0
            mask[mask != 0] = 1
            mask = mask.astype(np.uint8)
            struct = generate_binary_structure(3, 3)
            mask = binary_dilation(mask.copy(), structure=struct, iterations=3)
            mask = mask.astype(np.uint8)
            mask = from_nibabel(self.set_array(mask,verbose=False).nii)
        out = bc.n4_bias_field_correction(
            input_ants,
            mask=mask,
            shrink_factor=shrink_factor,
            convergence=convergence,
            spline_param=spline_param,
            verbose=verbose,
            weight_mask=weight_mask,
            
        )
        

        out_nib = out.to_nibabel()
        if crop:
            # Crop to regions that had a normalization applied. Removes a lot of dead space
            dif = NII((input_ants - out).to_nibabel())
            dif_arr = dif.get_array()
            dif_arr[dif_arr != 0] = 1
            dif.set_array_(dif_arr)
            ex_slice = dif.compute_crop_slice()
            out_nib = out_nib.slicer[ex_slice]
        
        if inplace:
            self.nii = out_nib
            self.set_dtype_(dtype)
            return self
        return NII(out_nib).set_dtype_(dtype)
    
    def n4_bias_field_correction_(self,threshold = 60,mask=None,shrink_factor=4,convergence={"iters": [50, 50, 50, 50], "tol": 1e-07},spline_param=200,verbose=False,weight_mask=None,crop=False):
        return self.n4_bias_field_correction(mask=mask,shrink_factor=shrink_factor,convergence=convergence,spline_param=spline_param,verbose=verbose,weight_mask=weight_mask,crop=crop,inplace=True,threshold = threshold)

    def match_histograms(self, reference:Image_Reference,c_val = 0,inplace=False):
        ref_nii = to_nii(reference)
        assert ref_nii.seg == False
        assert self.seg == False
        c_val = self.get_c_val(c_val)
        if c_val <= -999:
            raise ValueError('match_histograms only functions on MRI, which have a minimum 0.')

        from skimage.exposure import match_histograms as ski_match_histograms
        img_arr = self.get_array()
        matched = ski_match_histograms(img_arr, ref_nii.get_array())
        matched[matched <= c_val] = c_val        
        return self.set_array(matched, inplace=inplace)

    def match_histograms_(self, reference:Image_Reference,c_val = 0):
        return self.match_histograms(reference,c_val = c_val,inplace=True)

    def get_plane(self) -> str:
        """Determines the orientation plane of the NIfTI image along the x, y, or z-axis.

        Returns:
            str: The orientation plane of the image, which can be one of the following:
                - 'ax': Axial plane (along the z-axis).
                - 'cor': Coronal plane (along the y-axis).
                - 'sag': Sagittal plane (along the x-axis).
                - 'iso': Isotropic plane (if the image has equal zoom values along all axes).
        Examples:
            >>> nii = NII(nib.load('my_image.nii.gz'))
            >>> nii.get_plane()
            'ax'
        """
        plane_dict = {"S": "ax", "I": "ax", "L": "sag", "R": "sag", "A": "cor", "P": "cor"}
        img = to_nii(self)
        axc = np.array(nio.aff2axcodes(img.affine))
        zms = np.around(img.zoom, 1)
        ix_max = np.array(zms == np.amax(zms))
        num_max = np.count_nonzero(ix_max)
        if num_max == 2:
            plane = plane_dict[axc[~ix_max][0]]
        elif num_max == 1:
            plane = plane_dict[axc[ix_max][0]]
        else:
            plane = "iso"
        return plane

    #def erode_msk(self, mm: int = 5, connectivity: int = 3, inplace=False,verbose=True):
    #    """
    #    Erodes the binary segmentation mask by the specified number of voxels.
    #    Args:
    #        mm (int, optional): The number of voxels to erode the mask by. Defaults to 5.
    #        connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
    #        inplace (bool, optional): Whether to modify the mask in place or return a new object. Defaults to False.
    #        verbose (bool, optional): Whether to print a message indicating that the mask was eroded. Defaults to True.
    #    Returns:
    #        NII: The eroded mask.
    #    Notes:
    #        The method uses binary erosion with a 3D structuring element to erode the mask by the specified number of voxels.
    #    """
    #    log.print("erode mask",end='\r',verbose=verbose)
    #    msk_i_data = self.get_seg_array()
    #    out = np_erode_msk(msk_i_data, mm, connectivity)
    #    msk_e = Nifti1Image(out.astype(np.uint16), self.affine)
    #    if inplace:
    #        self.nii = msk_e
    #    log.print("Mask eroded by", mm, "voxels",verbose=verbose)
    #    return NII(msk_e,seg=True,c_val=0)
    #def erode_msk_(self, mm: int = 5, connectivity: int = 3, verbose=True):
    #    return self.erode_msk(mm=mm, connectivity=connectivity, inplace=True,verbose=True)
    #def dilate_msk_(self, mm:int = 5, connectivity: int=3, verbose=True):
    #    return self.dilate_msk(mm=mm, connectivity=connectivity, inplace=True, verbose=verbose)
    #
    #def dilate_msk(self, mm: int = 5, connectivity: int = 3, inplace=False, verbose=True):
    #    """
    #    Dilates the binary segmentation mask by the specified number of voxels.
    #
    #    Args:
    #        mm (int, optional): The number of voxels to dilate the mask by. Defaults to 5.
    #        connectivity (int, optional): Elements up to a squared distance of connectivity from the center are considered neighbors. connectivity may range from 1 (no diagonal elements are neighbors) to rank (all elements are neighbors).
    #        inplace (bool, optional): Whether to modify the mask in place or return a new object. Defaults to False.
    #        verbose (bool, optional): Whether to print a message indicating that the mask was dilated. Defaults to True.
    #
    #    Returns:
    #        NII: The dilated mask.
    #
    #    Notes:
    #        The method uses binary dilation with a 3D structuring element to dilate the mask by the specified number of voxels.
    #
    #    """
    #    log.print("dilate mask",end='\r',verbose=verbose)
    #    msk_i_data = self.get_seg_array()
    #    out = np_dilate_msk(msk_i_data, mm, connectivity)
    #    msk_e = Nifti1Image(out.astype(np.uint16), self.affine)
    #    if inplace:
    #        self.nii = msk_e
    #    log.print("Mask dilated by", mm, "voxels",verbose=verbose)
    #    return NII(msk_e,seg=True,c_val=0)
    #
    #def map_labels(self, label_map:Label_Map , verbose=True, inplace=False):
    #    """
    #    Maps labels in the given NIfTI image according to the label_map dictionary.
    #    Args:
    #        label_map (dict): A dictionary that maps the original label values (str or int) to the new label values (int).
    #            For example, `{"T1": 1, 2: 3, 4: 5}` will map the original labels "T1", 2, and 4 to the new labels 1, 3, and 5, respectively.
    #        verbose (bool): Whether to print the label mapping and the number of labels reassigned. Default is True.
    #        inplace (bool): Whether to modify the current NIfTI image object in place or create a new object with the mapped labels.
    #            Default is False.
    #    Returns:
    #        If inplace is True, returns the current NIfTI image object with mapped labels. Otherwise, returns a new NIfTI image object with mapped labels.
    #    """
    #    v_name2idx = {}
    #    data_orig = self.get_seg_array()
    #    labels_before = [v for v in np.unique(data_orig) if v > 0]
    #    # enforce keys to be str to support both str and int
    #    label_map_ = {
    #        (v_name2idx[k] if k in v_name2idx else int(k)): (
    #            v_name2idx[v] if v in v_name2idx else int(v)
    #        )
    #        for k, v in label_map.items()
    #    }
    #    log.print_dict(label_map_, "label_map_", verbose=verbose)
    #    data = np_map_labels(data_orig, label_map_)
    #    labels_after = [v for v in np.unique(data) if v > 0]
    #    log.print(
    #            "[*] N =",
    #            len(label_map_),
    #            "labels reassigned, before labels: ",
    #            labels_before,
    #            " after: ",
    #            labels_after,verbose=verbose
    #        )
    #    nii = Nifti1Image(data.astype(np.uint16), self.affine)
    #    if inplace:
    #        self.nii = nii
    #        return self
    #    return NII(nii, True)
    #
    #def map_labels_(self, label_map: Label_Map, verbose=True):
    #    return self.map_labels(label_map,verbose=verbose,inplace=True)
    def copy(self):
        return NII(Nifti1Image(self.get_array(), self.affine, self.header),seg=self.seg,c_val = self.c_val)
    def clone(self):
        return self.copy()
    def save(self,file:str|Path,make_parents=False,verbose=True):
        if make_parents:
            Path(file).parent.mkdir(exist_ok=True,parents=True)
        arr = self.get_array()
        out = Nifti1Image(arr, self.affine,self.header)#,dtype=arr.dtype)
        if self.seg:
            if arr.max()<256:
                out.set_data_dtype(np.uint8)
            elif arr.max()<65536:
                out.set_data_dtype(np.uint16)
            else:
                out.set_data_dtype(np.int32)
        print(f"Save {file} as {out.get_data_dtype()}") if verbose else None
        nib.save(out, file) #type: ignore
    def __str__(self) -> str:
        return f"shp={self.shape}; ori={self.orientation}, zoom={tuple(np.around(self.zoom, 5))}, seg={self.seg}"
    @classmethod
    def suppress_dtype_change_printout_in_set_array(cls, value=True):
        global suppress_dtype_change_printout_in_set_array
        suppress_dtype_change_printout_in_set_array = value
    def is_intersecting_vertical(self, b: Self, min_overlap=40) -> bool:
        '''
        Test if the image intersect in global space.
        assumes same Rotation
        TODO: Testing
        '''
        
        #warnings.warn('is_intersecting is untested use get_intersecting_volume instead')
        x1 = self.affine.dot([0, 0, 0, 1])[:3]
        x2 = self.affine.dot(self.shape + (1,))[:3]
        y1 = b.affine.dot([0, 0, 0, 1])[:3]
        y2 = b.affine.dot(b.shape + (1,))[:3]
        max_v = max(x1[2],x2[2])- min_overlap
        min_v = min(x1[2],x2[2])+ min_overlap
        if min_v < y1[2] < max_v:
            return True
        if min_v < y2[2] < max_v:
            return True

        max_v = max(y1[2],y2[2])- min_overlap
        min_v = min(y1[2],y2[2])+ min_overlap
        if min_v < x1[2] < max_v:
            return True
        if min_v < x2[2] < max_v:
            return True
        return False
    
    def get_intersecting_volume(self, b: Self) -> bool:
        '''
        computes intersecting volume
        '''
        b = b.copy()
        b.nii = Nifti1Image(b.get_array()*0+1,affine=b.affine)
        b.seg = True
        b.set_dtype_(np.uint8)
        b = b.resample_from_to(self,c_val=0,verbose=False)
        return b.get_array().sum()
        
    def extract_label(self,label:int, inplace=False):
        '''If this NII is a segmentation you can single out one label.'''
        assert label != 0, 'Zero label does not make sens. This is the background'
        seg_arr = self.get_seg_array()
        seg_arr[seg_arr != label] = 0
        seg_arr[seg_arr == label] = 1
        return self.set_array(seg_arr,inplace=inplace)
    def remove_labels(self,*label:int, inplace=False, verbose=True):
        '''If this NII is a segmentation you can single out one label.'''
        assert label != 0, 'Zero label does not make sens.  This is the background'
        seg_arr = self.get_seg_array()
        for l in label:
            seg_arr[seg_arr == l] = 0
        return self.set_array(seg_arr,inplace=inplace, verbose=verbose)
    
    def apply_mask(self,mask:Self, inplace=False):
        assert mask.shape == self.shape, f"[def apply_mask] Mask and Shape are not equal: \nMask - {mask},\nSelf - {self})"
        seg_arr = mask.get_seg_array()
        seg_arr[seg_arr != 0] = 1
        arr = self.get_array()
        return self.set_array(arr*seg_arr,inplace=inplace)

    def multiply(self,m:float, inplace=False):
        '''If this NII is a segmentation you can single out one label.'''
        warnings.warn("deprecated", DeprecationWarning)
        seg_arr = self.get_array()
        seg_arr*= m
        return self.set_array(seg_arr,inplace=inplace)
    
    def unique(self,verbose=False):
        '''Returns all integer labels WITHOUT 0. Must be performed only on a segmentation nii'''
        out = list(np.unique(self.get_seg_array()))
        print(out) if verbose else None
        return tuple(int(o) for o in out if o != 0)
        
def to_nii_optional(img_bids: Image_Reference|None, seg=False, default=None) -> NII | None:  # TODO
    if img_bids is None:
        return default
    try:
        return to_nii(img_bids,seg=seg)
    except ValueError:
        return default
    except KeyError:
        return default


def to_nii(img_bids: Image_Reference, seg=False) -> NII:  # TODO
    if isinstance(img_bids, NII):
        return img_bids.copy()
    #elif isinstance(img_bids, BIDS.bids_files.BIDS_FILE):
    #    return img_bids.open_nii()
    elif isinstance(img_bids, Path):
        return NII(nib.load(str(img_bids)), seg) #type: ignore
    elif isinstance(img_bids, str):
        return NII(nib.load(img_bids), seg) #type: ignore
    elif isinstance(img_bids, Nifti1Image): 
        return NII(img_bids, seg)
    else:
        raise ValueError(img_bids)

def to_nii_seg(img: Image_Reference) -> NII:
    return to_nii(img,seg=True)

def to_nii_interpolateable(i_img:Interpolateable_Image_Reference) -> NII:
    if isinstance(i_img,tuple):
        img, seg = i_img
        return to_nii(img,seg=seg)
    elif isinstance(i_img, NII):
        return i_img
    #elif isinstance(i_img,BIDS.bids_files.BIDS_FILE):
    #    return i_img.open_nii()
    else:
        raise ValueError("to_nii_interoplateable",i_img)
