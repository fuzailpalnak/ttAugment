import numpy as np
from imgaug.augmenters import meta

from tt_augment.tt_custom.tt_fwd_bkd import (
    MirrorFWD,
    MirrorBKD,
    FlipLR,
    FlipUD,
    RotateFWD,
    RotateBKD,
    ScaleFWD,
    ScaleBKD,
)


class TTCustom(meta.Augmenter):
    """
    Custom Augmenter are required for Geometric Transformation, When a geometric transformation is applied on the
    the test image the reverse has to be applied on the predicted image, to get the prediction as per the
    original non transformed image

    If an Image is rotated by 90, then the prediction is performed on the transformed image, but the prediction has to
    reversed to match the real image

    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()

        assert len(network_dimension) == 3, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (network_dimension,)
        )
        assert len(transform_dimension) == 3, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (transform_dimension,)
        )

        if network_dimension < transform_dimension:
            raise ValueError("Network Dimension Can't Be Less Than Transform Dimension")
        self.transform_dimension = transform_dimension
        self.network_dimension = network_dimension

    @property
    def reversal(self):
        return True

    def __call__(self, images, do_reversal=False):
        """

        :param images: batch images of dimension [Batch x Width x Height x Band]
        :param do_reversal: to reverse the augmentation
        :return:
        """
        if do_reversal:
            return self.bkd(images)
        else:
            return self.fwd(images)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def fwd(self, images: np.ndarray) -> np.ndarray:
        """
        This function applies transformation on the images which are about to be inferred, generally referred as
        test images

        :param images: numpy array images
        :return:
        """
        raise NotImplementedError

    def bkd(self, images: np.ndarray) -> np.ndarray:
        """
        This function applies transformation on the predicted images, The reason being, the geometric transformation
        applied on the test images have to be restored/reversed to get the back the original transformation.

        :param images:
        :return:
        """
        raise NotImplementedError


class Mirror(TTCustom):
    """
    Crop an image to transform_dimension and mirror the left pixel to match the size of network_dimension
    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)
        self.mirror_fwd = MirrorFWD(network_dimension, transform_dimension)
        self.mirror_bkd = MirrorBKD(network_dimension, transform_dimension)

    def fwd(self, images: np.ndarray) -> np.ndarray:
        return self.mirror_fwd(images=images)

    def bkd(self, images: np.ndarray) -> np.ndarray:
        return self.mirror_bkd(images=images)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]


class CropScale(TTCustom):
    """
    Crop an image to transform_dimension and rescale the image to match the size of network_dimension
    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)
        if transform_dimension >= network_dimension:
            raise ValueError("Can't Scale array with same Dimension")
        self.scale_fwd = ScaleFWD(network_dimension, transform_dimension)
        self.scale_bkd = ScaleBKD(network_dimension, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def fwd(self, images: np.ndarray) -> np.ndarray:
        return self.scale_fwd(images=images)

    def bkd(self, images: np.ndarray) -> np.ndarray:
        return self.scale_bkd(images=images)


class NoAugment(TTCustom):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)
        if network_dimension != transform_dimension:
            raise ValueError("Dimension Mis Match")

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def fwd(self, images: np.ndarray) -> np.ndarray:
        return images

    def bkd(self, images: np.ndarray) -> np.ndarray:
        return images


class Crop(NoAugment):
    """
    Crop an image to transform_dimension
    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)


class Rot(TTCustom):
    """
    Rotate an image
    """

    def __init__(self, angle: int, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

        self.rotate_fwd = RotateFWD(angle, transform_dimension)
        self.rotate_bkd = RotateBKD(angle, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def fwd(self, images: np.ndarray) -> np.ndarray:
        return self.rotate_fwd(images=images)

    def bkd(self, images: np.ndarray) -> np.ndarray:
        return self.rotate_bkd(images=images)


class FlipHorizontal(TTCustom):
    """
    Flip an image
    """

    def __init__(self, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

        self.flip = FlipLR(transform_dimension)

    def fwd(self, images: np.ndarray) -> np.ndarray:
        return self.flip(images=images)

    def bkd(self, images: np.ndarray) -> np.ndarray:
        return self.flip(images=images)


class FlipVertical(TTCustom):
    """
    Flip an image
    """

    def __init__(self, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

        self.flip = FlipUD(transform_dimension)

    def fwd(self, images: np.ndarray) -> np.ndarray:
        return self.flip(images=images)

    def bkd(self, images: np.ndarray) -> np.ndarray:
        return self.flip(images=images)
