from dataclasses import dataclass

from fragment.fragment import ImageFragment

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom, TTCustom

from sys import stdout


@dataclass
class FragmentTransformerCollect:
    name: str
    collection: list


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


class Mean:
    @staticmethod
    def collect(x, y):
        return x + y

    @staticmethod
    def aggregate(x, count):
        return x / count


class Transformer:
    def __init__(self, transformer, fragment):

        self._transformer = transformer
        self._data = None
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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def get_windowed_image(self, image: np.ndarray) -> np.ndarray:
        return self.fragment.get_fragment_data(image)


class TransformationType:
    def __init__(self, apply_per_image, inference):
        self.apply_per_image = apply_per_image
        self.inference = inference
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

    def run(self, image: np.ndarray):
        raise NotImplementedError

    def forward(self, transformer: Transformer, image: np.ndarray):
        raise NotImplementedError

    def reverse(self, transformer: Transformer, inferred_data):
        raise NotImplementedError

    def update(self, transformer: Transformer, reversed_data):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @staticmethod
    def generate_transformers(
        image_dimension: tuple, transformers: list, inference: tuple
    ):
        assert len(image_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        apply_per_image = list()
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
            apply_per_image.append(
                FragmentTransformerCollect(
                    name=transformer_name, collection=apply_per_fragment
                )
            )
        return apply_per_image


class Segmentation(TransformationType):
    def __init__(self, apply_per_image, inference):
        super().__init__(apply_per_image, inference)
        self.fragment_inference = None

    def run(self, image: np.ndarray):
        _, w, h, c = image.shape
        assert image.ndim == 4, (
            "Expected image to have shape (batch ,height, width, [channels]), "
            "got shape %s." % (image.shape,)
        )
        self.reset()
        for iterator, fragment_transformation in enumerate(self.apply_per_image):
            self.progress.Transformation_Count = "{}/{}".format(
                iterator + 1, len(self.apply_per_image)
            )
            for transformer_iterator, transformer in enumerate(
                fragment_transformation.collection
            ):
                self.progress.Transformer = "{}".format(fragment_transformation.name)
                self.progress.Fragment_Transformer_Progress = "{}/{}".format(
                    transformer_iterator + 1, len(fragment_transformation.collection)
                )

                transformer.data = transformer.get_windowed_image(image=image)
                Printer.print(self.progress.__dict__)

                yield transformer
            self.inference = Mean.collect(self.inference, self.fragment_inference)
        self.inference = Mean.aggregate(self.inference, len(self.apply_per_image))

    def forward(self, transformer: Transformer, image: np.ndarray):
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
        self.fragment_inference = transformer.fragment.transfer_fragment(
            transfer_from=reversed_data, transfer_to=self.fragment_inference
        )

    def reset(self):
        self.inference = np.zeros(self.inference.shape)
        self.fragment_inference = np.zeros(self.inference.shape)

    @classmethod
    def populate(cls, image_dimension: tuple, transformers: list, **kwargs):
        inference = kwargs["inference"]
        apply_per_image = cls.generate_transformers(
            image_dimension, transformers, inference
        )
        return cls(apply_per_image, inference)

    @classmethod
    def populate_binary(cls, image_dimension: tuple, transformers: list):
        assert len(image_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        batch, h, w, _ = image_dimension

        return cls.populate(
            image_dimension, transformers, inference=np.zeros((batch, h, w, 1))
        )

    @classmethod
    def populate_color(cls, image_dimension: tuple, transformers: list):
        assert len(image_dimension) == 4, (
            "Expected image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        batch, h, w, _ = image_dimension

        return cls.populate(
            image_dimension, transformers, inference=np.zeros((batch, h, w, 3))
        )


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
