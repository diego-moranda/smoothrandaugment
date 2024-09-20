import torch.nn.functional as F
import torch
from torch import Tensor
from smooth_geometric_augmentations import SmoothRotation, SmoothZoom, SmoothTranslateX, SmoothTranslateY, SmoothShearX, SmoothShearY
from smooth_color_augmentations import SmoothBrightness, SmoothContrast

class Smoothrandaugment(object):
    
    def __init__(self, num_ops=2, aug_p=1.0, linear=True, weighted_wave=False) -> None:
        """Smoothrandaugment augmentation

        Args:
            num_ops (int, optional): Number of augmentations to be randomly selected and applied. Defaults to 2.
            aug_p (float, optional): Augmentation probability for each augmentation. Defaults to 1.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
        """
        self.aug_p = aug_p
        self.num_ops = num_ops
        self.linear = linear
        self.weighted_wave = weighted_wave
        self.augmentation_space = self.__get_augmentation_space()
    
    def __get_augmentation_space(self) -> dict:
        """Get the augmentation space with specified parameters

        Returns:
            dict: Augmentation space dictionary
        """
        augmentation_space = {
            "Identity": (lambda vid: vid),
            "SmoothRotation": SmoothRotation(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave, multiple_augs=True),
            "SmoothZoom": SmoothZoom(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave, multiple_augs=True),
            "SmoothTranslateX": SmoothTranslateX(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave, multiple_augs=True),
            "SmoothTranslateY": SmoothTranslateY(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave, multiple_augs=True),
            "SmoothShearX": SmoothShearX(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave, multiple_augs=True),
            "SmoothShearY": SmoothShearY(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave, multiple_augs=True),
            "SmoothBrightness": SmoothBrightness(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave),
            "SmoothContrast": SmoothContrast(aug_p=self.aug_p, linear=self.linear, weighted_wave=self.weighted_wave)
        }
        
        return augmentation_space
    
    def __get_random_augmentation(self, dct:dict) -> tuple:
        """Get a random augmentation from the augmentation space

        Args:
            dct (dict): Augmentation space

        Returns:
            tuple: Augmentation ID, Augmentation
        """
        keys = tuple(dct.keys())
        key = keys[int(torch.randint(len(keys), ()))]
        
        return key, dct[key]
    
    def __call__(self, vid: Tensor) -> Tensor:
        
        for _ in range(self.num_ops):
            aug_id, augmentation = self.__get_random_augmentation(self.augmentation_space)
            vid = augmentation(vid)
            print(aug_id)
        
        return vid.transpose(1, 0)