import torch
from torch import Tensor
from smooth_augmentation import SmoothAugmentation
    
class SmoothColorAugmentation(SmoothAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=0.0, max_bound=1.0, linear=True, n_signals=50, weighted_wave=False) -> None:
        """Color augmentation base class.

        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum value for the augmentation. Defaults to 0.0.
            max_bound (float, optional): Maximum value for the augmentation. Defaults to 1.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave)
    
    
    def smooth_operation(self, vid:Tensor) -> Tensor:
        p = torch.rand(1)
        
        if p > self.aug_p:
            return vid
        
        vid = vid.unsqueeze(0) if len(vid.shape) < 5 else vid
        
        factors = self.gen_factor_func().unsqueeze(1).unsqueeze(1).unsqueeze(1)
        
        vid = self.color_function(vid, factors)
        
        return vid
    
    def color_function(self, vid:Tensor, factors:Tensor) -> Tensor:
        """Color augmenting function.

        Args:
            vid (Tensor): Video to augment
            factors (Tensor): Array of factors

        Returns:
            Tensor: Augmented video
        """
        pass
    
class SmoothBrightness(SmoothColorAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=-0.5, max_bound=0.5, linear=True, n_signals=50, weighted_wave=False) -> None:
        """Smooth Brightness augmentation.

        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum brightness value for the augmentation. Defaults to -0.5.
            max_bound (float, optional): Maximum brightness value for the augmentation. Defaults to 0.5.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave)
    
    def color_function(self, vid:Tensor, factors:Tensor) -> Tensor:
        return vid+factors

class SmoothContrast(SmoothColorAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=0.3, max_bound=1.5, linear=True, n_signals=50, weighted_wave=False) -> None:
        """Smooth Contrast augmentation.

        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum constrast value for the augmentation. Defaults to 0.3.
            max_bound (float, optional): Maximum constrast value for the augmentation. Defaults to 1.5.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave)
    
    def color_function(self, vid:Tensor, factors:Tensor) -> Tensor:
        return vid*factors