import cv2
import numpy as np
from imgaug import imresize_single_image

from imgaug.augmenters import Rotate, sm, meta
from imgaug.augmenters.flip import fliplr


class MirrorFWD(meta.Augmenter):
    """
    Mirror the pixel to get to network_dimension
    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()
        self.network_dimension = network_dimension
        self.transform_dimension = transform_dimension

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):

        images = batch.images
        nb_images = len(images)
        result = []
        for i in sm.xrange(nb_images):
            img = images[i]

            limit_w = (self.network_dimension[2] - self.transform_dimension[2]) // 2

            limit_h = (self.network_dimension[1] - self.transform_dimension[1]) // 2
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


class MirrorBKD(meta.Augmenter):
    """
    Remove the added pixel, Reverse of MirrorFWD
    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()
        self.network_dimension = network_dimension
        self.transform_dimension = transform_dimension

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        images = batch.images
        nb_images = len(images)
        result = []

        image_width, image_height = self.network_dimension[2], self.network_dimension[1]

        crop_width, crop_height = (
            self.transform_dimension[2],
            self.transform_dimension[1],
        )

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


class FlipLR(meta.Augmenter):
    """
    FLip an image
    """

    def __init__(self, transform_dimension: tuple):
        super().__init__()
        self.transform_dimension = transform_dimension

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i, images in enumerate(batch.images):
            batch.images[i] = fliplr(batch.images[i])
        return batch


class FlipUD(meta.Augmenter):
    """
    Flip an image
    """

    def __init__(self, transform_dimension: tuple):
        super().__init__()
        self.transform_dimension = transform_dimension

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i, images in enumerate(batch.images):
            batch.images[i] = batch.images[i][::-1, ...]
        return batch


class RotateFWD(meta.Augmenter):
    """
    Rotate an image with angle
    """

    def __init__(self, angle_axis: int, transform_dimension: tuple):
        super().__init__()
        self.transform_dimension = transform_dimension
        self.angle_axis = angle_axis

    def _augment_batch_(self, batch, random_state, parents, hooks):
        result = list()
        for i, images in enumerate(batch.images):
            image_rs = np.rot90(images, self.angle_axis)
            result.append(image_rs)
        batch.images = np.array(result, batch.images.dtype)
        return batch

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]


class RotateBKD(meta.Augmenter):
    """
    Rotate an image with -angle, Reverse the Rotate transformation
    """

    def __init__(self, angle_axis: int, transform_dimension: tuple):
        super().__init__()
        self.transform_dimension = transform_dimension
        self.angle_axis = angle_axis

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        result = list()
        for i, images in enumerate(batch.images):
            image_rs = np.rot90(images, -self.angle_axis)
            result.append(image_rs)
        batch.images = np.array(result, batch.images.dtype)
        return batch


class ScaleFWD(meta.Augmenter):
    """
    Scale an image from transform_dimension to network_dimension
    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()
        self.network_dimension = network_dimension
        self.transform_dimension = transform_dimension

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        result = list()
        for i, images in enumerate(batch.images):
            image_rs = imresize_single_image(
                images,
                (self.network_dimension[1], self.network_dimension[2]),
                interpolation="nearest",
            )
            result.append(image_rs)
        batch.images = np.array(result, batch.images.dtype)
        return batch


class ScaleBKD(meta.Augmenter):
    """
    Scale an image from network_dimension to transform_dimension
    """

    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()
        self.network_dimension = network_dimension
        self.transform_dimension = transform_dimension

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        result = list()
        for i, images in enumerate(batch.images):
            image_rs = imresize_single_image(
                images,
                (self.transform_dimension[1], self.transform_dimension[2]),
                interpolation="nearest",
            )
            result.append(image_rs)
        batch.images = np.array(result, batch.images.dtype)
        return batch


class NoSegAug(meta.Augmenter):
    def __init__(self, network_dimension: tuple, transform_dimension: tuple):
        super().__init__()
        self.network_dimension = network_dimension
        self.transform_dimension = transform_dimension

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        return batch
