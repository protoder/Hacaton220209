import random

import cv2
from matplotlib import pyplot as plt

import albumentations as A
import numpy as np

"""### Define a function to visualize images and masks"""


def visualize(image, mask, original_image=None, original_mask=None):
    Hori = np.concatenate((image, original_image, mask*255, original_mask*255), axis=1)

    # concatenate image Vertically
    cv2.imshow('', Hori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## HorizontalFlip


def MakeAug(image, mask, a = 9, verbose = False, p = 1):
    if a == 0:
        aug = A.HorizontalFlip(p=p)
    elif a == 1:
        aug = A.VerticalFlip(p=p)
    elif a == 2:
        aug = A.RandomRotate90(p=p)
    elif a == 3:
        aug = A.RandomRotate90(p=p)
    elif a == 4:
        aug = A.Transpose(p=p)
    elif a == 5:
        aug = A.ElasticTransform(p=p, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03)
    elif a == 6:
        aug = A.GridDistortion(p=p)
    elif a == 7:
        aug = A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1)
    elif a == 8:
        original_height, original_width = image.shape[:2]
        aug = A.RandomSizedCrop(min_max_height=(original_height*0.7, original_width*0.7), height=original_height, width=original_width, p=1)
    elif a == 9:
        original_height, original_width = image.shape[:2]
        aug = A.Compose([
            A.VerticalFlip(p=p/2),
            A.HorizontalFlip(p=p / 2),
            A.RandomRotate90(p=p/2),
            A.CLAHE(p=0.8),
            A.RandomGamma(p=0.8)])



    augmented = aug(image=image, mask=mask)

    image_h_flipped = augmented['image']
    mask_h_flipped = augmented['mask']

    if verbose:
        visualize(image_h_flipped, mask_h_flipped, original_image=image, original_mask=mask)

    return image_h_flipped, mask_h_flipped


