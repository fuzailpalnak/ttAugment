from imgaug.augmenters import meta

from tt_augment.tt_custom.custom import (
    MirrorFWD,
    MirrorBKD,
    FlipLR,
    FlipUD,
    RotateFWD,
    RotateBKD, ScaleFWD, ScaleBKD,
)


class TTFwdBkd(meta.Augmenter):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()

        if len(network_dimension) > 3 or len(transform_dimension) > 3:
            raise ValueError("Dimension MisMatch Expected WxHxB")
        if (
            network_dimension[0] < transform_dimension[0]
            or network_dimension[1] < transform_dimension[1]
        ):
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

    def fwd(self, images):
        raise NotImplementedError

    def bkd(self, images):
        raise NotImplementedError


class Mirror(TTFwdBkd):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)
        self.mirror_fwd = MirrorFWD(network_dimension, transform_dimension)
        self.mirror_bkd = MirrorBKD(network_dimension, transform_dimension)

    def fwd(self, images):
        return self.mirror_fwd(images=images)

    def bkd(self, images):
        return self.mirror_bkd(images=images)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]


class CropScale(TTFwdBkd):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)
        self.scale_fwd = ScaleFWD(network_dimension, transform_dimension)
        self.scale_bkd = ScaleBKD(network_dimension, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def fwd(self, images):
        return self.scale_fwd(images=images)

    def bkd(self, images):
        return self.scale_bkd(images=images)


class Crop(TTFwdBkd):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def fwd(self, images):
        pass

    def bkd(self, images):
        pass


class Rot(TTFwdBkd):
    def __init__(self, angle: int, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)
        self.rotate_fwd = RotateFWD(angle, transform_dimension)
        self.rotate_bkd = RotateBKD(angle, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def fwd(self, images):
        return self.rotate_fwd(images=images)

    def bkd(self, images):
        return self.rotate_bkd(images=images)


class FlipHorizontal(TTFwdBkd):
    def __init__(self, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

        self.flip = FlipLR(transform_dimension)

    def fwd(self, images):
        return self.flip(images=images)

    def bkd(self, images):
        return self.flip(images=images)


class FlipVertical(TTFwdBkd):
    def __init__(self, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

        self.flip = FlipUD(transform_dimension)

    def fwd(self, images):
        return self.flip(images=images)

    def bkd(self, images):
        return self.flip(images=images)
