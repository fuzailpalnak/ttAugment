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
    NoSegAug,
)


class TTCustom:
    """
    Custom Augmenter are required for Geometric Transformation, When a geometric transformation is applied on the
    the test image the reverse has to be applied on the predicted image, to get the prediction as per the
    original non transformed image

    If an Image is rotated by 90, then the prediction is performed on the transformed image, but the prediction has to
    reversed to match the real image

    """

    def __init__(
        self,
        fwd=None,
        seg_bkd=None,
        classification_bkd=None,
        network_dimension=None,
        transform_dimension=None,
    ):

        assert len(network_dimension) == 3, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (network_dimension,)
        )
        assert len(transform_dimension) == 3, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (transform_dimension,)
        )

        if fwd is not None:
            assert isinstance(fwd, meta.Augmenter), (
                "Expected to have fwd of type [meta.Augmenter], "
                "but received %s." % (type(fwd),)
            )
        if seg_bkd is not None:
            assert isinstance(seg_bkd, meta.Augmenter), (
                "Expected to have fwd of type [meta.Augmenter], "
                "but received %s." % (type(seg_bkd),)
            )
        if network_dimension < transform_dimension:
            raise ValueError("Network Dimension Can't Be Less Than Transform Dimension")
        self.transform_dimension = transform_dimension
        self.network_dimension = network_dimension

        self._fwd = fwd

        if seg_bkd is None:
            self._seg_bkd = NoSegAug(self.network_dimension, self.transform_dimension)
        else:
            self._seg_bkd = seg_bkd
        self._classification_bkd = classification_bkd

    @property
    def fwd(self) -> meta.Augmenter:
        """
        This function applies transformation on the images which are about to be inferred, generally referred as
        test images

        :return:
        """
        return self._fwd

    @property
    def seg_bkd(self) -> meta.Augmenter:
        """
        This function applies transformation on the predicted images, The reason being, the geometric transformation
        applied on the test images have to be restored/reversed to get the back the original transformation.

        :return:
        """
        return self._seg_bkd

    @property
    def classification_bkd(self):
        """
        Transform back the classification output

        :return:
        """
        return self._classification_bkd

    def get_parameters(self):
        return [self.network_dimension, self.transform_dimension]


class Mirror(TTCustom):
    """
    Crop an image to transform_dimension and mirror the left pixel to match the size of network_dimension
    """

    def __init__(self, transform_dimension: tuple, network_dimension: tuple):
        super().__init__(
            network_dimension=network_dimension, transform_dimension=transform_dimension
        )
        self._fwd = MirrorFWD(self.network_dimension, self.transform_dimension)
        self._seg = MirrorBKD(self.network_dimension, self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def seg_bkd(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_bkd(self):
        return self._classification


class CropScale(TTCustom):
    """
    Crop an image to transform_dimension and rescale the image to match the size of network_dimension
    """

    def __init__(self, transform_dimension: tuple, network_dimension: tuple):
        super().__init__(
            network_dimension=network_dimension, transform_dimension=transform_dimension
        )
        if transform_dimension >= network_dimension:
            raise ValueError("Can't Scale array with same Dimension")
        self._fwd = ScaleFWD(self.network_dimension, self.transform_dimension)
        self._seg = ScaleBKD(self.network_dimension, self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def seg_bkd(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_bkd(self):
        return self._classification


class NoAugment(TTCustom):
    def __init__(self, transform_dimension: tuple, network_dimension: tuple):
        super().__init__(
            network_dimension=network_dimension, transform_dimension=transform_dimension
        )
        if network_dimension != transform_dimension:
            raise ValueError("Dimension Mis Match")
        self._fwd = NoSegAug(self.network_dimension, self.transform_dimension)
        self._seg = NoSegAug(self.network_dimension, self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def seg_bkd(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_bkd(self):
        return self._classification


class Crop(NoAugment):
    """
    Crop an image to transform_dimension
    """

    def __init__(self, transform_dimension: tuple, network_dimension: tuple):
        super().__init__(
            network_dimension=network_dimension, transform_dimension=transform_dimension
        )


class Rot(TTCustom):
    """
    Rotate an image
    """

    def __init__(
        self, transform_dimension: tuple, network_dimension: tuple, angle: int
    ):
        super().__init__(
            network_dimension=network_dimension, transform_dimension=transform_dimension
        )

        self.angle = angle

        self._fwd = RotateFWD(self.angle, self.transform_dimension)
        self._seg = RotateBKD(self.angle, self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def seg_bkd(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_bkd(self):
        return self._classification


class FlipHorizontal(TTCustom):
    """
    Flip an image
    """

    def __init__(self, transform_dimension: tuple, network_dimension: tuple):
        super().__init__(
            network_dimension=network_dimension, transform_dimension=transform_dimension
        )

        self._fwd = FlipLR(self.transform_dimension)
        self._seg = FlipLR(self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def seg_bkd(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_bkd(self):
        return self._classification


class FlipVertical(TTCustom):
    """
    Flip an image
    """

    def __init__(self, transform_dimension: tuple, network_dimension: tuple):
        super().__init__(
            network_dimension=network_dimension, transform_dimension=transform_dimension
        )
        self._fwd = FlipUD(self.transform_dimension)
        self._seg = FlipUD(self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def seg_bkd(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_bkd(self):
        return self._classification
