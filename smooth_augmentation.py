import torch
from abc import ABC
from torch import Tensor

class SmoothAugmentation(ABC):
    
    def __init__(self, aug_p=1.0, n_frmaes=150, min_bound=0.0, max_bound=1.0, linear=True, n_signals=50, weighted_wave=False) -> None:
        """Base class for every smooth augmentation.

        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum value for the augmentation. Defaults to 0.0.
            max_bound (float, optional): Maximum value for the augmentation. Defaults to 1.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
        """
        
        self.aug_p = aug_p
        self.n_frames = n_frmaes
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.linear = linear
        self.n_signals = n_signals
        self.weighted_wave = weighted_wave
        
        self.gen_factor_func = self.__get_linear_factors if self.linear else self.__get_wave_factors
    
    def smooth_operation(self, vid: Tensor) -> Tensor:
        """Apply a smooth augmentation to a video.

        Args:
            vid (Tensor): Video to augment

        Returns:
            Tensor: Augmented video
        """
        pass
    
    def __get_linear_factors(self) -> Tensor:
        """Generates an array of evenly spaced factors between two random bounds.

        Returns:
            Tensor: Generated factors
        """
        min = torch.FloatTensor((1)).uniform_(self.min_bound, self.max_bound)
        max = torch.FloatTensor((1)).uniform_(self.min_bound, self.max_bound)
        factors = torch.linspace(min.item(), max.item(), self.n_frames)
        
        return factors
    
    def __get_wave_factors(self) -> Tensor:
        """Generates an array of factors from a normalized sum of sinusoidals used by the augmentations.

        Returns:
            Tensor: Generated factors
        """
        max_val = self.max_bound - self.min_bound
        amplitude = torch.FloatTensor((self.n_signals)).uniform_(-1., 1.)
        wavelength = torch.FloatTensor((self.n_signals)).uniform_(1.5, 4.)
        velocity = 1
        time = torch.FloatTensor((self.n_signals)).uniform_(-4., 4.)
        
        x = torch.linspace(-1, 1, self.n_frames).repeat(self.n_signals).reshape((self.n_signals, self.n_frames)).T
        
        waves = amplitude * torch.sin((2*torch.pi/wavelength) * (x - time))
        
        res_wave = waves.sum(1)
        res_wave = max_val * (res_wave - res_wave.min())/(res_wave.max()-res_wave.min()) + self.min_bound
        
        w = torch.FloatTensor((1)).uniform_(0.5, 1) if self.weighted_wave else 1.0

        return w*res_wave    
    
    def __call__(self, vid: Tensor) -> Tensor:
        return self.smooth_operation(vid)