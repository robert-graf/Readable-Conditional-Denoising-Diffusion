import operator
from math import ceil, floor
from numbers import Number

import numpy as np
from typing_extensions import Self

# fmt: off

class NII_Proxy:
    pass
C = Self|Number|np.ndarray
class NII_Math(NII_Proxy):

    def _binary_opt(self, other:C, opt,inplace = False)-> Self:
        if isinstance(other,NII_Math):
            other = other.get_array()
        return self.set_array(opt(self.get_array(),other),inplace=inplace)
    def _uni_opt(self, opt,inplace = False)-> Self:
        return self.set_array(opt(self.get_array()),inplace=inplace)
    def __add__(self,p2):
        return self._binary_opt(p2,operator.add)
    def __sub__(self,p2):
        return self._binary_opt(p2,operator.sub)
    def __mul__(self,p2):
        return self._binary_opt(p2,operator.mul)
    def __pow__(self,p2):
        return self._binary_opt(p2,operator.pow)
    def __truediv__(self,p2):
        return self._binary_opt(p2,operator.truediv)
    def __floordiv__(self,p2):
        return self._binary_opt(p2,operator.floordiv)
    def __mod__(self,p2):
        return self._binary_opt(p2,operator.mod)
    def __lshift__(self,p2):
        return self._binary_opt(p2,operator.lshift)
    def __rshift__(self,p2):
        return self._binary_opt(p2,operator.rshift)
    def __and__(self,p2):
        return self._binary_opt(p2,operator.add)
    def __or__(self,p2):
        return self._binary_opt(p2,operator.or_)
    def __xor__(self,p2):
        return self._binary_opt(p2,operator.xor)
    def __invert__(self):
        return self._uni_opt(operator.invert)

    def __lt__(self,p2):
        return self._binary_opt(p2,operator.lt)
    def __le__(self,p2):
            return self._binary_opt(p2,operator.le)
    def __eq__(self,p2):
            return self._binary_opt(p2,operator.eq)
    def __ne__(self,p2):
            return self._binary_opt(p2,operator.ne)
    def __gt__(self,p2):
            return self._binary_opt(p2,operator.gt)
    def __ge__(self,p2):
            return self._binary_opt(p2,operator.ge)

    def __iadd__(self,p2):
        return self._binary_opt(p2,operator.add,inplace=True)
    def __isub__(self,p2:C):
        return self._binary_opt(p2,operator.sub,inplace=True)
    def __imul__(self,p2):
        return self._binary_opt(p2,operator.mul,inplace=True)
    def __ipow__(self,p2):
        return self._binary_opt(p2,operator.pow,inplace=True)
    def __itruediv__(self,p2):
        return self._binary_opt(p2,operator.truediv,inplace=True)
    def __ifloordiv__(self,p2):
        return self._binary_opt(p2,operator.floordiv,inplace=True)
    def __imod__(self,p2):
        return self._binary_opt(p2,operator.mod,inplace=True)

    def __neg__(self):
        return self._uni_opt(operator.neg)
    def __pos__(self):
        return self._uni_opt(operator.pos)
    def __abs__(self):
        return self._uni_opt(operator.abs)

    def __round__(self):
        return self._uni_opt(np.round)
    def __floor__(self):
        return self._uni_opt(np.floor)
    def __ceil__(self):
        return self._uni_opt(np.ceil)
    def max(self)->float:
        return self.get_array().max()
    def min(self)->float:
        return self.get_array().min()

    def clamp(self, min=None,max=None,inplace=False)->Self:
        arr = self.get_array()
        if min != None:
            arr[arr<= min] = min
        if max != None:
            arr[arr>= max] = max
        return self.set_array(arr,inplace=inplace)

    def clamp_(self, min=None,max=None):
        return self.clamp(min,max,inplace=True)

    def normalize(self,min_out = 0, max_out = 1, quantile = 1., clamp_lower:float|None=None,inplace=False):
        max_v = np.quantile(self.get_array(),q=quantile)
        arr = self.clamp(clamp_lower,max_v,inplace=inplace)
        arr -= arr.min() - min_out/max_out
        arr /= arr.max() *max_out
        #TODO TEST
        assert arr.max() == max_out, max_out
        assert arr.min() == min_out, min_out
        return self
    def normalize_(self,min_out = 0, max_out = 1, quantile = 1., clamp_lower:float|None=None):
        return self.normalize(min_out = min_out, max_out = max_out, quantile = quantile, clamp_lower=clamp_lower,inplace=True)
    def normalize_mri(self,min_out = 0, max_out = 1,inplace=False):
        return self.normalize(min_out = min_out, max_out = max_out, quantile = .99, clamp_lower=0,inplace=inplace)
    def normalize_ct(self,min_out = 0, max_out = 1,inplace=False):
        arr = self.clamp(min=-1024,max=1024,inplace=inplace)
        return arr.normalize(min_out = min_out, max_out = max_out, inplace=inplace)
    def pad_to(self,target_shape:list[int]|tuple[int,int,int], mode="constant",inplace = False):
        padding = []
        crop = []
        for in_size, out_size in zip(self.shape[-3:], target_shape[-3:], strict=False):
            to_pad_size = max(0, out_size - in_size) / 2.0
            to_crop_size = -min(0, out_size - in_size) / 2.0
            padding.extend([(ceil(to_pad_size), floor(to_pad_size))])
            if to_crop_size == 0:
                crop.append(slice(None))
            else:
                end = -floor(to_crop_size)
                if end == 0:
                    end = None
                crop.append(slice(ceil(to_crop_size), end))
        assert len(self.shape) == 3, f"TODO add >3 dim support: {self.shape}"
        x_ = np.pad(self.get_array()[tuple(crop)],padding,mode=mode,constant_values=self.get_c_val())
        return self.set_array(x_,inplace=inplace)
    def sum(self,axis = None,keepdims=False,where = np._NoValue):
        if hasattr(where,"get_array"):
            where=where.get_array().astype(bool)

        return np.sum(self.get_array(),axis=axis,keepdims=keepdims,where=where)
    def threshold(self,threshold=0.5, inplace=False):
        arr = self.get_array()
        arr2 = arr.copy()
        arr[arr2>=threshold] = 1
        arr[arr2<=threshold] = 0
        nii = self.set_array(arr,inplace)
        nii.seg =True
        nii.c_val = 0
        return nii
    def ssim(self, nii:NII_Proxy, minV = 0):
        from skimage.metrics import structural_similarity as ssim
        img_1 = nii.get_array() - minV
        img_2 = self.get_array() - minV
        img_1/= img_1.max()
        img_1[img_1<=0] = 0
        img_2/= img_2.max()
        img_2[img_2<=0] = 0
        ssim_value = ssim(img_1, img_2,data_range=img_1.max() - img_1.min())
        return ssim_value
