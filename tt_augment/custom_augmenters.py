from imgaug.augmenters import meta


class RescaleCrop(meta.Augmenter):
    def __init__(self):
        super().__init__()
        self.do_reversal = False

    def __call__(self, images, do_reversal=False):
        self.do_reversal = do_reversal
        return self.augment(images=images)

    def get_parameters(self):
        pass

    def _augment_batch_(self, batch, random_state, parents, hooks):
        pass

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):

        pass


class MirrorCrop(meta.Augmenter):
    def __init__(self, transform_dimension: tuple):
        super().__init__()
        self.do_reversal = False
        self.dimension = transform_dimension

    def __call__(self, images, do_reversal=False):
        self.do_reversal = do_reversal
        return self.augment(images=images)

    def get_parameters(self):
        pass

    def _augment_batch_(self, batch, random_state, parents, hooks):
        pass

    def _augment_images_by_samples(
        self, images, samples, image_shapes=None, return_matrices=False
    ):

        pass
