from tt_augment import tt_custom, arithmetic_aggregate
from tt_augment.printer import Printer

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom, TTCustom
from tt_augment.window import Window


class Sibling:
    def __init__(self, transformer, window):

        self._transformer = transformer
        self._window = window
        self._name = self.transformer.__class__.__name__

    @property
    def name(self):
        return self._name

    @property
    def window(self):
        return self._window

    @property
    def transformer(self):
        return self._transformer

    def data(self, image: np.ndarray) -> np.ndarray:
        return image[
            :,
            self.window[0][0] : self.window[0][1],
            self.window[1][0] : self.window[1][1],
            :,
        ]


class Member(list):
    def __init__(self):
        super().__init__()

    def add_sibling(self, sibling: Sibling):
        self.append(sibling)

    def get_sibling(self):
        for sibling in self:
            yield sibling


class FamilyProgress:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Family(list):
    def __init__(self):
        super().__init__()

        self._tt_inferred_family = None
        self._tt_inferred_member = None

        self._name = self.__class__.__name__
        self.family_progress = FamilyProgress(**{"Family": self._name, "FamilyCount": "None",  "Sibling": "None",
                           "Sibling_Progress": "None", "Infer": list()})

    @property
    def name(self):
        return self._name

    @property
    def tt_inferred_family(self):
        return self._tt_inferred_family

    @tt_inferred_family.setter
    def tt_inferred_family(self, value):
        self._tt_inferred_family = value

    @property
    def tt_inferred_member(self):
        return self._tt_inferred_member

    @tt_inferred_member.setter
    def tt_inferred_member(self, value):
        self._tt_inferred_member = value

    def add_member(self, member: Member):
        raise NotImplementedError

    def add_inference(self, inferred_data, sibling):
        raise NotImplementedError

    def tt(self, image: np.ndarray, arithmetic_compute: str):
        raise NotImplementedError

    def tt_fwd(self, image: np.ndarray, sibling: Sibling) -> np.ndarray:
        raise NotImplementedError

    def tt_bkd(self, inferred_data, sibling: Sibling):
        raise NotImplementedError


class Segmentation(Family, list):
    def __init__(self, members=None):
        super().__init__()
        if members is None:
            list.__init__(self, [])
        elif isinstance(members, Iterable):
            assert all([isinstance(member, Member) for member in members]), (
                "Expected all children to be augmenters, got types %s."
                % (", ".join([str(type(v)) for v in members]))
            )
            list.__init__(self, members)
        else:
            raise Exception(
                "Expected None or Member or list of Members, "
                "got %s." % (type(members),)
            )

    def add_member(self, member: Member):
        assert isinstance(
            member, Member
        ), "Expected member to be Member, got types %s." % (type(member),)
        self.append(member)

    def tt_fwd(self, image: np.ndarray, sibling: Sibling) -> np.ndarray:
        _, w, h, c = image.shape
        assert image.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (image.shape,)
        )

        assert sibling.transformer.transform_dimension == (w, h, c), (
            "Expected image to have shape (%s), "
            "got shape %s." % (sibling.transformer.transform_dimension, (w, h, c),)
        )

        return sibling.transformer.fwd(
            images=image
        )

    def tt_bkd(self, inferred_data, sibling: Sibling):
        assert inferred_data.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (inferred_data.shape,)
        )

        assert isinstance(
            sibling, Sibling
        ), "Expected child to be Member, got types %s." % (str(type(sibling)))

        return sibling.transformer.segmentation_reverse(images=inferred_data)

    def tt(self, image: np.ndarray, arithmetic_compute="mean"):
        _, w, h, c = image.shape
        assert image.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (image.shape,)
        )

        arithmetic_calculation = getattr(arithmetic_aggregate, arithmetic_compute)()
        self.tt_inferred_family = np.zeros(image.shape)

        for member_iterator, member in enumerate(self):
            self.family_progress.FamilyCount = "{}/{}".format(member_iterator + 1, len(self))
            Printer.print(self.family_progress.__dict__)

            self.tt_inferred_member = np.zeros(image.shape)

            for iterator, sibling in enumerate(member.get_sibling()):
                self.family_progress.Sibling = sibling.name
                self.family_progress.Sibling_Progress = "{}/{}".format(iterator + 1, len(member))
                Printer.print(self.family_progress.__dict__)

                yield sibling.data(image=image), sibling
            infer_done = self.family_progress.Infer
            infer_done.append(self.family_progress.Sibling)
            self.family_progress.Infer = infer_done

            Printer.print(self.family_progress.__dict__)

            self.tt_inferred_family = arithmetic_calculation.collect(
                self.tt_inferred_family, self.tt_inferred_member
            )
        self._tt_inferred_family = arithmetic_calculation.aggregate(self.tt_inferred_family, len(self))

    def add_inference(self, inferred_data: np.ndarray, sibling: Sibling):

        part_1_x = sibling.window[0][0]
        part_1_y = sibling.window[0][1]
        part_2_x = sibling.window[1][0]
        part_2_y = sibling.window[1][1]

        cropped_image = self.tt_inferred_member[:, part_1_x:part_1_y, part_2_x:part_2_y, :]

        inferred_image = cropped_image + inferred_data

        if np.any(cropped_image):
            intersecting_inference_elements = np.zeros(cropped_image.shape)
            intersecting_inference_elements[cropped_image > 0] = 1

            non_intersecting_inference_elements = 1 - intersecting_inference_elements

            intersected_inference = inferred_image * intersecting_inference_elements
            aggregate_inference = intersected_inference / 2

            non_intersected_inference = np.multiply(
                non_intersecting_inference_elements, inferred_image
            )
            inferred_image = aggregate_inference + non_intersected_inference
        self.tt_inferred_member[:, part_1_x:part_1_y, part_2_x:part_2_y, :] = inferred_image


class Classification(Family, list):
    def __init__(self):
        super().__init__()

    def add_member(self, member: Member):
        raise NotImplementedError

    def add_inference(self, inferred_data, sibling: Sibling):
        raise NotImplementedError

    def tt(self, image: np.ndarray, arithmetic_compute: str):
        pass

    def tt_fwd(self, image: np.ndarray, sibling: Sibling) -> np.ndarray:
        pass

    def tt_bkd(self, inferred_data, sibling: Sibling):
        pass


def generate_family_members(image_dimension: tuple, transformers: list):
    assert len(image_dimension) == 3, (
        "Expected image to have shape (width, height, [channels]), "
        "got shape %s." % (image_dimension,)
    )

    family = list()

    for individual_transformer in transformers:
        member = Member()

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
        window = Window.get_window(
            window_size=transformer.transform_dimension, org_size=image_dimension,
        )

        for win_number, win in window:
            member.add_sibling(Sibling(transformer=transformer, window=win))
        family.append(member)
    return family


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
        custom_aug = getattr(tt_custom, transformer_name)(**transformer_param)
        return TTCustom(
            fwd=custom_aug,
            network_dimension=network_dimension,
            transform_dimension=transform_dimension,
        )
    else:
        raise Exception("UnSupported Transformer %s." % (transformer_name,))


def segmentation_tta(image_dimension: tuple, transformers: list):
    family_members = generate_family_members(image_dimension, transformers)
    return Segmentation(family_members)


def classification_tta(image_dimension: tuple, transformers: list):
    raise NotImplementedError
