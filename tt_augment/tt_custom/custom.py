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
        segmentation_reverse=None,
        classification_reverse=None,
        inference_dimension=None,
        transform_dimension=None,
    ):

        assert len(inference_dimension) == 4, (
            "Expected image to have shape (batch, height, height, [channels]), "
            "got shape %s." % (inference_dimension,)
        )
        assert len(transform_dimension) == 4, (
            "Expected image to have shape (batch, height, height, [channels]), "
            "got shape %s." % (transform_dimension,)
        )

        if fwd is not None:
            assert isinstance(
                fwd, meta.Augmenter
            ), "Expected to have fwd of type [meta.Augmenter], " "but received %s." % (
                type(fwd),
            )
        if segmentation_reverse is not None:
            assert isinstance(
                segmentation_reverse, meta.Augmenter
            ), "Expected to have fwd of type [meta.Augmenter], " "but received %s." % (
                type(segmentation_reverse),
            )
        if inference_dimension < transform_dimension:
            raise ValueError(
                "Inference Dimension Can't Be Less Than Transform Dimension"
            )
        self.transform_dimension = transform_dimension
        self.inference_dimension = inference_dimension

        self._fwd = fwd

        if segmentation_reverse is None:
            self._segmentation_reverse = NoSegAug(
                self.inference_dimension, self.transform_dimension
            )
        else:
            self._segmentation_reverse = segmentation_reverse
        self._classification_reverse = classification_reverse

    @property
    def fwd(self) -> meta.Augmenter:
        """
        This function applies transformation on the images which are about to be inferred, generally referred as
        test images

        :return:
        """
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        """
        This function applies transformation on the predicted images, The reason being, the geometric transformation
        applied on the test images have to be restored/reversed to get the back the original transformation.

        :return:
        """
        return self._segmentation_reverse

    @property
    def classification_reverse(self):
        """
        Transform back the classification output

        :return:
        """
        return self._classification_reverse

    def get_parameters(self):
        return [self.inference_dimension, self.transform_dimension]


class Mirror(TTCustom):
    """
    Crop an image to transform_dimension and mirror the left pixel to match the size of inference_dimension
    """

    def __init__(self, transform_dimension: tuple, inference_dimension: tuple):
        super().__init__(
            inference_dimension=inference_dimension,
            transform_dimension=transform_dimension,
        )
        self._fwd = MirrorFWD(self.inference_dimension, self.transform_dimension)
        self._seg = MirrorBKD(self.inference_dimension, self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class CropScale(TTCustom):
    """
    Crop an image to transform_dimension and rescale the image to match the size of inference_dimension
    """

    def __init__(self, transform_dimension: tuple, inference_dimension: tuple):
        super().__init__(
            inference_dimension=inference_dimension,
            transform_dimension=transform_dimension,
        )
        if transform_dimension >= inference_dimension:
            raise ValueError("Can't Scale array with same Dimension")
        self._fwd = ScaleFWD(self.inference_dimension, self.transform_dimension)
        self._seg = ScaleBKD(self.inference_dimension, self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class NoAugment(TTCustom):
    def __init__(self, transform_dimension: tuple, inference_dimension: tuple):
        assert inference_dimension == transform_dimension, (
            "While Using NoAugment Transformation ",
            "Expected [inference_dimension] and [transform_dimension] to be equal",
            "got %s and %s",
            (transform_dimension, inference_dimension),
        )
        super().__init__(
            inference_dimension=inference_dimension,
            transform_dimension=transform_dimension,
        )
        if inference_dimension != transform_dimension:
            raise ValueError("Dimension Mis Match")
        self._fwd = NoSegAug(self.inference_dimension, self.transform_dimension)
        self._seg = NoSegAug(self.inference_dimension, self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class Crop(NoAugment):
    """
    Crop an image to transform_dimension
    """

    def __init__(self, transform_dimension: tuple, inference_dimension: tuple):
        super().__init__(
            inference_dimension=inference_dimension,
            transform_dimension=transform_dimension,
        )


class Rot(TTCustom):
    """
    Rotate an image
    """

    def __init__(
        self, transform_dimension: tuple, inference_dimension: tuple, angle: int
    ):
        assert inference_dimension == transform_dimension, (
            "While Using Geometric Transformation ",
            "Expected [inference_dimension] and [transform_dimension] to be equal",
            "got %s and %s",
            (transform_dimension, inference_dimension),
        )
        super().__init__(
            inference_dimension=inference_dimension,
            transform_dimension=transform_dimension,
        )

        angle_axis = [0, 90, 180, 270]
        assert angle in angle_axis, (
            "Expected angle to be [0,  90, 180, 270]",
            "given %s",
            (angle,),
        )

        self._fwd = RotateFWD(angle_axis.index(angle), self.transform_dimension)
        self._seg = RotateBKD(angle_axis.index(angle), self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class FlipHorizontal(TTCustom):
    """
    Flip an image
    """

    def __init__(self, transform_dimension: tuple, inference_dimension: tuple):
        assert inference_dimension == transform_dimension, (
            "While Using Geometric Transformation ",
            "Expected [inference_dimension] and [transform_dimension] to be equal",
            "got %s and %s",
            (transform_dimension, inference_dimension),
        )
        super().__init__(
            inference_dimension=inference_dimension,
            transform_dimension=transform_dimension,
        )

        self._fwd = FlipLR(self.transform_dimension)
        self._seg = FlipLR(self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification


class FlipVertical(TTCustom):
    """
    Flip an image
    """

    def __init__(self, transform_dimension: tuple, inference_dimension: tuple):
        assert inference_dimension == transform_dimension, (
            "While Using Geometric Transformation ",
            "Expected [inference_dimension] and [transform_dimension] to be equal",
            "got %s and %s",
            (transform_dimension, inference_dimension),
        )
        super().__init__(
            inference_dimension=inference_dimension,
            transform_dimension=transform_dimension,
        )
        self._fwd = FlipUD(self.transform_dimension)
        self._seg = FlipUD(self.transform_dimension)
        self._classification = None

    @property
    def fwd(self) -> meta.Augmenter:
        return self._fwd

    @property
    def segmentation_reverse(self) -> meta.Augmenter:
        return self._seg

    @property
    def classification_reverse(self):
        return self._classification
