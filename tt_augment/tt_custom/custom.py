import cv2
import numpy as np

from imgaug.augmenters import Rotate, sm, meta
from imgaug.augmenters.flip import fliplr


class TTCustom(meta.Augmenter):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()

        if len(network_dimension) > 3 or len(transform_dimension) > 3:
            raise ValueError("Dimension MisMatch Expected WxHxB")
        if (
            network_dimension[0] < transform_dimension[0]
            or network_dimension[1] < transform_dimension[1]
        ):
            raise ValueError("Network Dimension Can't Be Less Than Transform Dimension")
        self.dimension = transform_dimension
        self.network_dimension = network_dimension

    def get_parameters(self):
        return [self.network_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        pass


class MirrorFWD(TTCustom):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):

        images = batch.images
        nb_images = len(images)
        result = []
        for i in sm.xrange(nb_images):
            img = images[i]

            limit_w = (self.network_dimension[0] - self.dimension[0]) // 2

            limit_h = (self.network_dimension[1] - self.dimension[1]) // 2
            img = cv2.copyMakeBorder(
                img,
                limit_h,
                limit_h,
                limit_w,
                limit_w,
                borderType=cv2.BORDER_REFLECT_101,
            )
            result.append(img)
        batch.images = np.array(result, images.dtype)
        return batch


class MirrorBKD(TTCustom):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_dimension, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        images = batch.images
        nb_images = len(images)
        result = []

        image_width, image_height = self.network_dimension[0], self.network_dimension[1]

        crop_width, crop_height = self.dimension[0], self.dimension[1]

        for i in sm.xrange(nb_images):
            img = images[i]

            dy = (image_height - crop_height) // 2
            dx = (image_width - crop_width) // 2

            y1 = dy
            y2 = y1 + crop_height
            x1 = dx
            x2 = x1 + crop_width

            result.append(img[y1:y2, x1:x2, :])
        batch.images = np.array(result, images.dtype)
        return batch


class FlipLR(TTCustom):
    def __init__(self, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i, images in enumerate(batch.images):
            batch.images[i] = fliplr(batch.images[i])
        return batch


class FlipUD(TTCustom):
    def __init__(self, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

    def get_parameters(self):
        return [self.network_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i, images in enumerate(batch.images):
            batch.images[i] = batch.images[i][::-1, ...]
        return batch


class RotateFWD(TTCustom):
    def __init__(self, angle: int, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

        self.transform = Rotate(rotate=angle)

    def __call__(self, images):
        return self.transform.augment(images=images)

    def get_parameters(self):
        return [self.network_dimension, self.dimension]


class RotateBKD(TTCustom):
    def __init__(self, angle: int, transform_dimension: tuple):
        super().__init__(transform_dimension, transform_dimension)

        self.transform = Rotate(rotate=-angle)

    def __call__(self, images):
        return self.transform.augment(images=images)

    def get_parameters(self):
        return [self.network_dimension, self.dimension]
