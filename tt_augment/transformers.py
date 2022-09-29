from typing import Dict, List

from dataclasses import dataclass

from tt_augment import tt_custom

from image_fragment.fragment import ImageFragment
from collections import defaultdict

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom, TTCustom

from py_oneliner import one_liner


@dataclass
class TransformationOnImage:
    name: str
    collection: list


class TransformationsPerRun(list):
    def __init__(self, merge_transformations="mean", transformation_on_image=None):
        if transformation_on_image is None:
            list.__init__(self, [])
        elif isinstance(transformation_on_image, TransformationOnImage):
            list.__init__(self, [transformation_on_image])
        elif isinstance(transformation_on_image, Iterable):
            assert all(
                [
                    isinstance(per_image, TransformationOnImage)
                    for per_image in transformation_on_image
                ]
            ), (
                "Expected all children to be FragmentTransformationPerImage, got types %s."
                % (", ".join([str(type(v)) for v in transformation_on_image]))
            )
            list.__init__(self, transformation_on_image)
        else:
            raise Exception(
                "Expected None or FragmentTransformationPerImage or list of"
                " FragmentTransformationPerImage, "
                "got %s." % (type(transformation_on_image),)
            )

        assert merge_transformations in ["mean"], (
            "Expected merge_transformations to be in ['mean']",
            "got %s",
            (merge_transformations,),
        )
        self.merge_transformations = merge_transformations
        self.output = None

    def add(self, transformation_on_image: TransformationOnImage):
        self.append(transformation_on_image)

    def collect(self, individual_transformation_output):
        if self.output is None:
            self.output = individual_transformation_output

        else:
            if self.merge_transformations == "mean":
                self.output = self.output + individual_transformation_output

    def aggregate(self):
        if self.merge_transformations == "mean":
            self.output = self.output / len(self)


class Transformer:
    def __init__(self, transformer, fragment):

        self._transformer = transformer
        self._name = self.transformer.__class__.__name__
        self._fragment = fragment

        self._output = None

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def fragment(self):
        return self._fragment

    @property
    def name(self):
        return self._name

    @property
    def transformer(self):
        return self._transformer

    def get_windowed_image(self, image: np.ndarray) -> np.ndarray:
        return self.fragment.get_fragment_data(image)


class ImageTransformation:
    def __init__(self, transformation_to_apply: List[Dict]):
        self._transformation_to_apply = transformation_to_apply
        self._fragment_data = defaultdict(list)

        self._image = None
        self._cached_transformation = None

        self.transformation_output = None

    def _create_transformation_fragments(
        self, image_dimension: tuple, window_dimension: tuple
    ):
        if self._image != image_dimension:
            transformations = self.generate_transformers(
                image_dimension,
                window_dimension,
                self._transformation_to_apply,
            )
            self._cached_transformation = transformations

    def transformations_fragments(
        self, image_dimension: tuple, window_dimension: tuple, output_dimension: tuple
    ):
        self._create_transformation_fragments(image_dimension, window_dimension)

        for iterator, transformation in enumerate(self._cached_transformation):
            self.transformation_output = self._create_transformation_image(
                dim=output_dimension
            )

            one_liner.one_line(
                tag="Transformations",
                tag_data=f"{iterator+1}/{len(self._cached_transformation)}",
                tag_color="cyan",
                tag_data_color="yellow",
            )

            for transformer_iterator, transformer in enumerate(
                transformation.collection
            ):

                one_liner.one_line(
                    tag=f"{transformation.name}",
                    tag_data=f"{transformer_iterator+1}/{len(transformation.collection)}",
                    tag_color="cyan",
                    tag_data_color="yellow",
                )

                yield transformer

            if self.transformation_output is not None:
                self._cached_transformation.collect(self.transformation_output)

    def apply_transformation(self, transformer: Transformer, image: np.ndarray):
        raise NotImplementedError

    def append(self, transformer: Transformer, inferred_data):
        raise NotImplementedError

    def _create_transformation_image(self, dim: tuple):
        raise NotImplementedError

    def tta_output(self):
        self._cached_transformation.aggregate()
        self.transformation_output = None

        return self._cached_transformation.output

    @staticmethod
    def generate_transformers(
        image_dimension: tuple, window_dimension: tuple, transformers: list
    ):
        assert len(image_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        assert len(window_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (window_dimension,)
        )

        transformations = TransformationsPerRun()
        for individual_transformer in transformers:
            transformer_name = individual_transformer["name"]

            if "crop_to_dimension" not in list(individual_transformer.keys()):
                crop_to_dimension = window_dimension
            else:
                crop_to_dimension = individual_transformer["crop_to_dimension"]

            if "param" not in list(individual_transformer.keys()):
                transformer_param = {}
            else:
                transformer_param = individual_transformer["param"]

            transformer = look_up(
                transformer_name,
                crop_to_dimension,
                window_dimension,
                **transformer_param,
            )
            if transformer.crop_to_dimension > image_dimension:
                raise ValueError(
                    "Transformation Dimension Can't be bigger that Image Dimension"
                )
            fragments = ImageFragment.image_fragment_4d(
                fragment_size=transformer.crop_to_dimension, org_size=image_dimension
            )
            apply_per_fragment = list()
            for fragment in fragments:
                apply_per_fragment.append(
                    Transformer(transformer=transformer, fragment=fragment)
                )

            transformations.add(
                transformation_on_image=TransformationOnImage(
                    name=transformer_name, collection=apply_per_fragment
                )
            )

        return transformations


class Segmentation(ImageTransformation):
    def __init__(self, transformation_to_apply: List[Dict]):
        super().__init__(transformation_to_apply)

    def apply_transformation(self, transformer: Transformer, image: np.ndarray):
        assert image.ndim == 4, (
            "Expected image to have shape (batch ,height, width, [channels]), "
            "got shape %s." % (image.shape,)
        )

        assert isinstance(
            transformer, Transformer
        ), "Expected child to be Transformer, got types %s." % (str(type(transformer)))

        return transformer.transformer.fwd(
            images=transformer.get_windowed_image(image=image)
        )

    def append(self, transformer: Transformer, inferred_data: np.ndarray):
        assert inferred_data.ndim == 4, (
            "Expected image to have shape (batch ,height, width, [channels]), "
            "got shape %s." % (inferred_data.shape,)
        )

        assert isinstance(
            transformer, Transformer
        ), "Expected child to be Transformer, got types %s." % (str(type(transformer)))

        restored_image = transformer.transformer.segmentation_reverse(
            images=inferred_data
        )
        transformer.output = restored_image

        self.transformation_output = transformer.fragment.transfer_fragment(
            transfer_from=restored_image, transfer_to=self.transformation_output
        )

    def _create_transformation_image(self, dim: tuple):
        return np.zeros(dim)


def look_up(transformer_name, crop_to_dimension, window_dimension, **transformer_param):
    if hasattr(custom, transformer_name):
        custom_aug = getattr(custom, transformer_name)(
            crop_to_dimension=crop_to_dimension,
            window_dimension=window_dimension,
            **transformer_param,
        )
        return custom_aug
    elif hasattr(tt_custom, transformer_name):
        assert window_dimension == crop_to_dimension, (
            "While Using External Color Augmentation ",
            "Expected [window_dimension] and [crop_to_dimension] to be equal",
            "got %s and %s",
            (crop_to_dimension, window_dimension),
        )
        custom_aug = getattr(tt_custom, transformer_name)(**transformer_param)
        return TTCustom(
            fwd=custom_aug,
            window_dimension=crop_to_dimension,
            crop_to_dimension=crop_to_dimension,
        )
    else:
        raise Exception("UnSupported Transformer %s." % (transformer_name,))
