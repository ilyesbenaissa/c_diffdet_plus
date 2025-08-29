import numpy as np
import random
from detectron2.data import transforms as T
from fvcore.transforms.transform import Transform

class BrightnessContrastTransform(Transform):
    """
    A transform that adjusts brightness and contrast to help with dent detection
    under different lighting conditions.
    """
    def __init__(self, brightness_factor, contrast_factor):
        super().__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """
        Apply brightness and contrast adjustment to the image
        
        Args:
            img (ndarray): of shape HxWxC, RGB
        
        Returns:
            ndarray: the augmented image, of shape HxWxC
        """
        # Apply brightness adjustment
        img = img.astype(np.float32)
        img = img * self.brightness_factor
        
        # Apply contrast adjustment
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * self.contrast_factor + mean
        
        # Clip to valid range
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Do nothing for the coordinates
        """
        return coords
    
    def inverse(self):
        """
        The inverse is not implemented.
        """
        return T.NoOpTransform()

class RandomBrightnessContrast(T.Augmentation):
    """
    Randomly adjusts brightness and contrast to help with dent detection
    under different lighting conditions.
    
    This is especially useful for detecting dents that may be obscured by highlights
    or when scratches are mixed with dents.
    """
    def __init__(self, brightness_delta=0.3, contrast_delta=0.3, prob=0.5):
        """
        Args:
            brightness_delta (float): Maximum brightness adjustment range
            contrast_delta (float): Maximum contrast adjustment range
            prob (float): Probability of applying the augmentation
        """
        super().__init__()
        self.brightness_delta = brightness_delta
        self.contrast_delta = contrast_delta
        self.prob = prob
        
    def get_transform(self, image):
        if random.random() > self.prob:
            return T.NoOpTransform()
            
        brightness_factor = np.random.uniform(1-self.brightness_delta, 1+self.brightness_delta)
        contrast_factor = np.random.uniform(1-self.contrast_delta, 1+self.contrast_delta)
        
        return BrightnessContrastTransform(brightness_factor, contrast_factor)