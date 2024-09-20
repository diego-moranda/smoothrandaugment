# Smoothrandaugment

The `Smoothrandaugment` algorithm provides a novel approach on video augmentations. The fundamental concept is analogous to the [RandAugment](https://arxiv.org/abs/1909.13719) augmentation, yet this novel methodology employs dynamic augmentations that evolve over time. The benefits of using dynamic augmentations on video data have already been demonstrated in [[1]](#1).

This augmentation was mainly designed for thermal video and used to improve the robustness of the model developed in [[2]](#2) for the [HIRA](https://www.hira-project.com) project.

Before using the smooth augmentations, a visual inspection of the augmented video is raccomended, in order to fine-tune the bounds for a specific task.

## Smooth Geometric augmentations
There are 4 kind of smooth geometric augmentations:
- Rotation
- Zoom
- Translation on X and Y
- Shear on X and Y

All these augmentations have a `multi_augs` boolean parameter that must be set to `True` when two or more augmentations are used, for example in the [Compose](https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html) fuction.

## Smooth Color augmentations
There are 2 color intesity based augmentations:
- Brightness
- Contrast

The aforementioned augmentations merely alter the color intensity of the video frames in a dynamic manner.

## References
<a id=1>[1]</a>
Taeoh Kim et al. Exploring temporally dynamic data augmentation for video recognition, 2022.

<a id=2>[2]</a>
Federica Gioia, Filippo Pura, Marco Forgione, Dario Piga, Alberto Greco, and Arcangelo Merla. Respiratory frequency reconstruction from thermal video signals: an end- to-end deep learning approach. In preparation, 2024.
