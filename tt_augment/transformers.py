from dataclasses import dataclass

from tt_augment import tt_custom

from fragment.fragment import ImageFragment

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom, TTCustom

from sys import stdout


@dataclass
class TransformationOnImage:
    name: str
    collection: list
    blue_print: tuple


class Progress:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Printer:
    @staticmethod
    def print(data):
        if type(data) == dict:
            data = Printer.dict_to_string(data)
        stdout.write("\r\033[1;37m>>\x1b[K" + data.__str__())
        stdout.flush()

    @staticmethod
    def print_new_line(data):
        stdout.write("\n")
        stdout.write("\r\033[1;37m>>\x1b[K" + data.__str__())
        stdout.flush()

    @staticmethod
    def dict_to_string(input_dict, separator=", "):
        combined_list = list()
        for key, value in input_dict.items():
            individual = "{} : {}".format(key, value)
            combined_list.append(individual)
        return separator.join(combined_list)


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

        if self.merge_transformations == "mean":
            self.output = self.output + individual_transformation_output

    def aggregate(self):
        if self.merge_transformations == "mean":
            self.output = self.output / len(self)


class InferenceStore:
    def __init__(self, count: int, merge_type="mean"):
        self.merge_type = merge_type
        self.count = count
        self.inference = None

    def collect(self, fragment_inference):
        if self.inference is None:
            self.inference = fragment_inference

        if self.merge_type == "mean":
            self.inference = self.inference + fragment_inference

    def aggregate(self):
        if self.merge_type == "mean":
            self.inference = self.inference / self.count


class Transformer:
    def __init__(self, transformer, fragment):

        self._transformer = transformer
        self._name = self.transformer.__class__.__name__
        self._fragment = fragment

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


class TransformationType:
    def __init__(self, transformations: TransformationsPerRun):
        self.transformations = transformations
        self.transformation_output = None
        self.progress = Progress(
            **{
                "Name": self.__class__.__name__,
                "Transformation_Count": "None",
                "Transformer": "None",
                "Fragment_Transformer_Progress": "None",
            }
        )

    @classmethod
    def populate(cls, image_dimension: tuple, transformers: list, **kwargs):
        pass

    def run(self):
        for iterator, transformation in enumerate(self.transformations):
            self.transformation_output = np.zeros(transformation.blue_print)

            self.progress.Transformation_Count = "{}/{}".format(
                iterator + 1, len(self.transformations)
            )
            for transformer_iterator, transformer in enumerate(
                transformation.collection
            ):
                self.progress.Transformer = "{}".format(transformation.name)
                self.progress.Fragment_Transformer_Progress = "{}/{}".format(
                    transformer_iterator + 1, len(transformation.collection)
                )

                Printer.print(self.progress.__dict__)

                yield transformer
            self.transformations.collect(self.transformation_output)
        self.transformations.aggregate()

    def forward(self, transformer: Transformer, image: np.ndarray):
        raise NotImplementedError

    def reverse(self, transformer: Transformer, inferred_data):
        raise NotImplementedError

    def update(self, transformer: Transformer, reversed_data):
        raise NotImplementedError

    @staticmethod
    def generate_transformers(
        image_dimension: tuple, transformers: list, blue_print: tuple
    ):
        assert len(image_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image_dimension,)
        )
        transformations = TransformationsPerRun()
        for individual_transformer in transformers:
            assert list(individual_transformer.keys()) == [
                "name",
                "param",
                "transform_dimension",
                "network_dimension",
            ], "Expected Keys ['name', 'param', 'transform_dimension', 'network_dimension'], "

            transformer_name, transformer_param = (
                individual_transformer["name"],
                individual_transformer["param"],
            )
            transform_dimension = individual_transformer["transform_dimension"]
            network_dimension = individual_transformer["network_dimension"]

            transformer = look_up(
                transformer_name,
                transform_dimension,
                network_dimension,
                **transformer_param
            )
            if transformer.transform_dimension > image_dimension:
                raise ValueError(
                    "Transformation Dimension Can't be bigger that Image Dimension"
                )
            fragments = ImageFragment.image_fragment_4d(
                fragment_size=transformer.transform_dimension, org_size=image_dimension
            )
            apply_per_fragment = list()
            for fragment in fragments:
                apply_per_fragment.append(
                    Transformer(transformer=transformer, fragment=fragment)
                )

            transformations.add(
                transformation_on_image=TransformationOnImage(
                    name=transformer_name,
                    collection=apply_per_fragment,
                    blue_print=blue_print,
                )
            )

        return transformations


class Segmentation(TransformationType):
    def __init__(self, transformations: TransformationsPerRun):
        super().__init__(transformations)

    def forward(self, transformer: Transformer, image: np.ndarray):
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

    def reverse(self, transformer: Transformer, inferred_data: np.ndarray):
        assert inferred_data.ndim == 4, (
            "Expected image to have shape (batch ,height, width, [channels]), "
            "got shape %s." % (inferred_data.shape,)
        )

        assert isinstance(
            transformer, Transformer
        ), "Expected child to be Transformer, got types %s." % (str(type(transformer)))

        return transformer.transformer.segmentation_reverse(images=inferred_data)

    def update(self, transformer: Transformer, reversed_data: np.ndarray):
        assert reversed_data.ndim == 4, (
            "Expected image to have shape (batch ,height, width, [channels]), "
            "got shape %s." % (reversed_data.shape,)
        )
        self.transformation_output = transformer.fragment.transfer_fragment(
            transfer_from=reversed_data, transfer_to=self.transformation_output
        )

    @classmethod
    def populate(cls, image_dimension: tuple, transformers: list, **kwargs):
        blue_print = kwargs["blue_print"]
        transformations = cls.generate_transformers(
            image_dimension, transformers, blue_print
        )

        return cls(transformations)

    @classmethod
    def populate_binary(cls, image_dimension: tuple, transformers: list):
        assert len(image_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        batch, h, w, _ = image_dimension

        return cls.populate(image_dimension, transformers, blue_print=(batch, h, w, 1))

    @classmethod
    def populate_color(cls, image_dimension: tuple, transformers: list):
        assert len(image_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        batch, h, w, _ = image_dimension
        return cls.populate(image_dimension, transformers, blue_print=(batch, h, w, 3))


def look_up(
    transformer_name, transform_dimension, network_dimension, **transformer_param
):
    if hasattr(custom, transformer_name):
        custom_aug = getattr(custom, transformer_name)(
            transform_dimension=transform_dimension,
            network_dimension=network_dimension,
            **transformer_param
        )
        return custom_aug
    elif hasattr(tt_custom, transformer_name):
        assert network_dimension == transform_dimension, (
            "While Using External Color Augmentation ",
            "Expected [network_dimension] and [transform_dimension] to be equal",
            "got %s and %s",
            (transform_dimension, network_dimension),
        )
        custom_aug = getattr(tt_custom, transformer_name)(**transformer_param)
        return TTCustom(
            fwd=custom_aug,
            network_dimension=network_dimension,
            transform_dimension=transform_dimension,
        )
    else:
        raise Exception("UnSupported Transformer %s." % (transformer_name,))
