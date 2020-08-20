from imgaug.augmenters import meta, Rotate
from imgaug.augmenters.flip import fliplr


class TTCustom(meta.Augmenter):
    def __init__(self, network_input_dimension: tuple, transform_dimension: tuple):
        super().__init__()
        self.dimension = transform_dimension
        self.network_input_dimension = network_input_dimension

    @property
    def reversal(self):
        return True

    def __call__(self, images, do_reversal=False):
        return self.augment(images=images)

    def get_parameters(self):
        return [self.network_input_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        pass

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):
        pass


class ScaleAdjust(TTCustom):
    def __init__(self, network_input_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_input_dimension, transform_dimension)

    def __call__(self, images, do_reversal=False):
        self.do_reversal = do_reversal
        return self.augment(images=images)

    def get_parameters(self):
        return [self.network_input_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        pass

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):

        pass


class CropAdjust(TTCustom):
    def __init__(self, network_input_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_input_dimension, transform_dimension)

    def __call__(self, images, do_reversal=False):
        return self.augment(images=images)

    def get_parameters(self):
        return [self.network_input_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        pass

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):
        pass


class MirrorAdjust(TTCustom):
    def __init__(self, network_input_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_input_dimension, transform_dimension)

    def __call__(self, images, do_reversal=False):
        return self.augment(images=images)

    def get_parameters(self):
        return [self.network_input_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        pass

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):

        pass


class FlipLR(TTCustom):
    def __init__(self, network_input_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_input_dimension, transform_dimension)

    def __call__(self, images, do_reversal=False):
        return self.augment(images=images)

    def get_parameters(self):
        return [self.network_input_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i, images in enumerate(batch.images):
            batch.images[i] = fliplr(batch.images[i])
        return batch

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):
        pass


class FlipUD(TTCustom):
    def __init__(self, network_input_dimension: tuple, transform_dimension: tuple):
        super().__init__(network_input_dimension, transform_dimension)

    def __call__(self, images, do_reversal=False):
        return self.augment(images=images)

    def get_parameters(self):
        return [self.network_input_dimension, self.dimension]

    def _augment_batch_(self, batch, random_state, parents, hooks):
        for i, images in enumerate(batch.images):
            batch.images[i] = batch.images[i][::-1, ...]
        return batch

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):
        pass


class Rot(TTCustom):
    def __init__(
        self, angle: int, network_input_dimension: tuple, transform_dimension: tuple
    ):
        super().__init__(network_input_dimension, transform_dimension)

        self.transform = Rotate(rotate=angle)
        self.reverse_transform = Rotate(rotate=-angle)

    def __call__(self, images, do_reversal=False):
        self.do_reversal = do_reversal
        if do_reversal:
            return self.reverse_transform(images=images)
        else:
            return self.transform(images=images)

    def get_parameters(self):
        return [self.network_input_dimension, self.dimension]
