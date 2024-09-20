import torch.nn.functional as F
import torch
from torch import Tensor
from smooth_augmentation import SmoothAugmentation
    
class SmoothGeometricAugmentation(SmoothAugmentation):
    
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=0.0, max_bound=1.0, linear=True, n_signals=50, weighted_wave=False, multiple_augs=False) -> None:
        """Geometric augmentation base class.
        
        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum value for the augmentation. Defaults to 0.0.
            max_bound (float, optional): Maximum value for the augmentation. Defaults to 1.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
            multiple_augs (bool, optional): Set this to true when chaining multiple geometric augmentations. Don't set to True for the last augmentation. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave)
        self.multiple_augs = multiple_augs
        
    def smooth_operation(self, vid) -> Tensor:
        
        p = torch.rand(1)
        
        if p > self.aug_p:
            return vid
        
        vid = vid.unsqueeze(0) if len(vid.shape) < 5 else vid
        
        factors = self.gen_factor_func()
        
        aff_mat = self.get_affine_matrix(factors)
        
        grid = F.affine_grid(aff_mat, vid[0].size(), align_corners=True)
        vid = F.grid_sample(vid[0], grid, padding_mode="zeros", align_corners=True)
        
        return vid if self.multiple_augs else vid.transpose(1, 0)
    
    def get_affine_matrix(self, factors: Tensor) -> Tensor:
        """Get the affine matrix used by a geometric augmentation.

        Args:
            factors (Tensor): Generated factors used by the affine matrix

        Returns:
            Tensor: Affine matrix
        """
        pass

class SmoothRotation(SmoothGeometricAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=-30.0, max_bound=30.0, linear=True, n_signals=50, weighted_wave=False, multiple_augs=False) -> None:
        """Smooth Rotation.
    
        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum angle in degrees for the augmentation. Defaults to -30.0.
            max_bound (float, optional): Maximum angle in degrees the augmentation. Defaults to 30.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
            multiple_augs (bool, optional): Set this to true when chaining multiple geometric augmentations. Don't set to True for the last augmentation. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave, multiple_augs)
        
    def get_affine_matrix(self, factors: Tensor) -> Tensor:
        angles = factors / 180 * torch.pi
        s = torch.sin(angles)
        c = torch.cos(angles)
        
        aff_mat = torch.stack((torch.stack([c, -s, torch.zeros(self.n_frames)], dim=1),
                            torch.stack([s, c, torch.zeros(self.n_frames)], dim=1)), dim=1)
        return aff_mat

class SmoothZoom(SmoothGeometricAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=0.5, max_bound=2.0, linear=True, n_signals=50, weighted_wave=False, multiple_augs=False) -> None:
        """Smooth Zoom.
    
        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum zoom factor for the augmentation. Defaults to 0.5.
            max_bound (float, optional): Maximum zoom factor the augmentation. Defaults to 2.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
            multiple_augs (bool, optional): Set this to true when chaining multiple geometric augmentations. Don't set to True for the last augmentation. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave, multiple_augs)
    
    def get_affine_matrix(self, factors:Tensor) -> Tensor:
        aff_mat = torch.stack((torch.stack([torch.ones(self.n_frames)/factors, torch.zeros(self.n_frames), torch.zeros(self.n_frames)], dim=1),
                            torch.stack([torch.zeros(self.n_frames), torch.ones(self.n_frames)/factors, torch.zeros(self.n_frames)], dim=1)), dim=1)
        return aff_mat
    
class SmoothTranslateX(SmoothGeometricAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=-1.0, max_bound=1.0, linear=True, n_signals=50, weighted_wave=False, multiple_augs=False) -> None:
        """Smooth Translate on X axis.
    
        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum translate factor for the augmentation. Defaults to -1.0.
            max_bound (float, optional): Maximum translate factor the augmentation. Defaults to 1.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
            multiple_augs (bool, optional): Set this to true when chaining multiple geometric augmentations. Don't set to True for the last augmentation. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave, multiple_augs)
    
    def get_affine_matrix(self, factors:Tensor) -> Tensor:
        aff_mat = torch.stack((torch.stack([torch.ones(self.n_frames), torch.zeros(self.n_frames), factors], dim=1),
                            torch.stack([torch.zeros(self.n_frames), torch.ones(self.n_frames), torch.zeros(self.n_frames)], dim=1)), dim=1)
        return aff_mat

class SmoothTranslateY(SmoothGeometricAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=-1.0, max_bound=1.0, linear=True, n_signals=50, weighted_wave=False, multiple_augs=False) -> None:
        """Smooth Translate on Y axis.
    
        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum translate factor for the augmentation. Defaults to -1.0.
            max_bound (float, optional): Maximum translate factor the augmentation. Defaults to 1.0.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
            multiple_augs (bool, optional): Set this to true when chaining multiple geometric augmentations. Don't set to True for the last augmentation. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave, multiple_augs)
    
    def get_affine_matrix(self, factors:Tensor) -> Tensor:
        aff_mat = torch.stack((torch.stack([torch.ones(self.n_frames), torch.zeros(self.n_frames), torch.zeros(self.n_frames)], dim=1),
                            torch.stack([torch.zeros(self.n_frames), torch.ones(self.n_frames), factors], dim=1)), dim=1)
        return aff_mat
    
class SmoothShearX(SmoothGeometricAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=-0.5, max_bound=0.5, linear=True, n_signals=50, weighted_wave=False, multiple_augs=False) -> None:
        """Smooth Shear on X axis.
    
        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum shear factor for the augmentation. Defaults to -0.5.
            max_bound (float, optional): Maximum shear factor the augmentation. Defaults to 0.5.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
            multiple_augs (bool, optional): Set this to true when chaining multiple geometric augmentations. Don't set to True for the last augmentation. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave, multiple_augs)
    
    def get_affine_matrix(self, factors:Tensor) -> Tensor:
        aff_mat = torch.stack((torch.stack([torch.ones(self.n_frames), factors, torch.zeros(self.n_frames)], dim=1),
                            torch.stack([torch.zeros(self.n_frames), torch.ones(self.n_frames), torch.zeros(self.n_frames)], dim=1)), dim=1)
        return aff_mat
    
class SmoothShearY(SmoothGeometricAugmentation):
    def __init__(self, aug_p=1, n_frmaes=150, min_bound=-0.5, max_bound=0.5, linear=True, n_signals=50, weighted_wave=False, multiple_augs=False) -> None:
        """Smooth Shear on Y axis.
    
        Args:
            aug_p (float, optional): Agumentation probability. Defaults to 1.0.
            n_frmaes (int, optional): Number of frames of the videos. Defaults to 150.
            min_bound (float, optional): Minimum shear factor for the augmentation. Defaults to -0.5.
            max_bound (float, optional): Maximum shear factor the augmentation. Defaults to 0.5.
            linear (bool, optional): Whether to use linear factors (True) or generated from sine waves (False). Defaults to True.
            n_signals (int, optional): Number of sinusoidal signals to be used when using non linear technique (linear=False). Defaults to 50.
            weighted_wave (bool, optional): Whether to use a random weight that divides the generated factors of the sinusoidal technique. Defaults to False.
            multiple_augs (bool, optional): Set this to true when chaining multiple geometric augmentations. Don't set to True for the last augmentation. Defaults to False.
        """
        super().__init__(aug_p, n_frmaes, min_bound, max_bound, linear, n_signals, weighted_wave, multiple_augs)
    
    def get_affine_matrix(self, factors:Tensor) -> Tensor:
        aff_mat = torch.stack((torch.stack([torch.ones(self.n_frames), torch.zeros(self.n_frames), torch.zeros(self.n_frames)], dim=1),
                            torch.stack([factors, torch.ones(self.n_frames), torch.zeros(self.n_frames)], dim=1)), dim=1)
        return aff_mat