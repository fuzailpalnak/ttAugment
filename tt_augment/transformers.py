from fragment.fragment import ImageFragment

from tt_augment import tt_custom, arithmetic_aggregate

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom, TTCustom

from sys import stdout


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


class TTInfo:
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


class Member(list):
    def __init__(self):
        super().__init__()

    def add_image_info(self, tt_info: TTInfo):
        self.append(tt_info)

    def get_image_info(self):
        for tt_info in self:
            yield tt_info


class FamilyProgress:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Family(list):
    def __init__(self, members: list, family_data, member_data, arithmetic_calculation):
        super().__init__()
        self.members = members
        self.arithmetic_calculation = arithmetic_calculation

        self._tt_family_data = family_data
        self._tt_member_data = member_data

        self._name = self.__class__.__name__
        self.family_progress = FamilyProgress(
            **{
                "Family": self._name,
                "FamilyCount": "None",
                "Sibling": "None",
                "Sibling_Progress": "None",
            }
        )

    @property
    def name(self):
        return self._name

    @property
    def tt_family_data(self):
        return self._tt_family_data

    @tt_family_data.setter
    def tt_family_data(self, value):
        self._tt_family_data = value

    @property
    def tt_member_data(self):
        return self._tt_member_data

    @tt_member_data.setter
    def tt_member_data(self, value):
        self._tt_member_data = value

    def add_inference(self, tt_info: TTInfo):
        raise NotImplementedError

    def tt(self, image: np.ndarray):
        _, w, h, c = image.shape
        assert image.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (image.shape,)
        )

        for member_iterator, member in enumerate(self.members):
            self.family_progress.FamilyCount = "{}/{}".format(
                member_iterator + 1, len(self.members)
            )
            Printer.print(self.family_progress.__dict__)

            for iterator, tt_info in enumerate(member.get_image_info()):
                self.family_progress.Sibling = tt_info.name
                self.family_progress.Sibling_Progress = "{}/{}".format(
                    iterator + 1, len(member)
                )
                Printer.print(self.family_progress.__dict__)
                tt_info.data = tt_info.get_windowed_image(image=image)
                yield tt_info
            Printer.print(self.family_progress.__dict__)

            self.tt_family_data = self.arithmetic_calculation.collect(
                self.tt_family_data, self.tt_member_data
            )
        self.tt_family_data = self.arithmetic_calculation.aggregate(
            self.tt_family_data, len(self.members)
        )

    def tt_fwd(self, tt_info: TTInfo) -> np.ndarray:
        return tt_info.transformer.fwd(images=tt_info.data)

    def tt_bkd(self, inferred_data, tt_info: TTInfo) -> TTInfo:
        raise NotImplementedError


class Segmentation(Family):
    def __init__(
        self,
        members: list,
        family_data: np.ndarray,
        member_data: np.ndarray,
        arithmetic_calculation,
    ):
        super().__init__(members, family_data, member_data, arithmetic_calculation)

    def tt_bkd(self, inferred_data, tt_info: TTInfo) -> TTInfo:
        assert inferred_data.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (inferred_data.shape,)
        )

        assert isinstance(
            tt_info, TTInfo
        ), "Expected child to be Member, got types %s." % (str(type(tt_info)))

        tt_info.data = tt_info.transformer.segmentation_reverse(images=inferred_data)
        return tt_info

    def add_inference(self, tt_info: TTInfo):
        self._tt_member_data = tt_info.fragment.transfer_fragment(
            transfer_from=tt_info.data, transfer_to=self._tt_member_data
        )


class Classification(Family):
    def __init__(self, members, family_data, member_data, arithmetic_calculation):
        super().__init__(members, family_data, member_data, arithmetic_calculation)

    def add_inference(self, tt_info: TTInfo):
        raise NotImplementedError

    def tt_fwd(self, tt_info: TTInfo) -> np.ndarray:
        pass

    def tt_bkd(self, inferred_data, tt_info: TTInfo):
        pass


def generate_family_members(image_dimension: tuple, transformers: list):
    assert len(image_dimension) == 4, (
        "Expected image to have shape (batch, height, width, [channels]), "
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
        fragments = ImageFragment.image_fragment_4d(
            fragment_size=transformer.transform_dimension, org_size=image_dimension
        )
        for fragment in fragments:
            member.add_image_info(TTInfo(transformer=transformer, fragment=fragment))
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


def segmentation_binary(
    image_dimension: tuple,
    transformers: list,
    arithmetic_compute="mean",
):
    assert len(image_dimension) == 4, (
        "Expected image to have shape (batch, height, width, [channels]), "
        "got shape %s." % (image_dimension,)
    )

    assert arithmetic_compute in [
        "mean",
        "geometric_mean",
    ], "Expected values ['mean', 'geometric_mean'], " "got %s." % (arithmetic_compute,)
    batch, h, w, _ = image_dimension

    if arithmetic_compute == "mean":
        inference_data = np.zeros((batch, h, w, 1))
    else:
        inference_data = np.ones((batch, h, w, 1))
    arithmetic_calculation = getattr(arithmetic_aggregate, arithmetic_compute)()
    family_members = generate_family_members(image_dimension, transformers)
    return Segmentation(
        members=family_members,
        family_data=inference_data,
        member_data=np.zeros(inference_data.shape),
        arithmetic_calculation=arithmetic_calculation,
    )


def segmentation_color(
    image_dimension: tuple,
    transformers: list,
    arithmetic_compute="mean",
):
    assert (
        len(image_dimension) == 4
    ), "Expected image to have shape (height, width, [channels]), " "got shape %s." % (
        image_dimension,
    )

    assert arithmetic_compute in [
        "mean",
        "geometric_mean",
    ], "Expected values ['mean', 'geometric_mean'], " "got %s." % (arithmetic_compute,)
    batch, h, w, _ = image_dimension

    if arithmetic_compute == "mean":
        inference_data = np.zeros((batch, h, w, 3))
    else:
        inference_data = np.ones((batch, h, w, 3))
    arithmetic_calculation = getattr(arithmetic_aggregate, arithmetic_compute)()
    family_members = generate_family_members(image_dimension, transformers)
    return Segmentation(
        members=family_members,
        family_data=inference_data,
        member_data=np.zeros(inference_data.shape),
        arithmetic_calculation=arithmetic_calculation,
    )


def classification(
    image_dimension: tuple,
    batch_size: int,
    transformers: list,
    arithmetic_compute="mean",
):
    raise NotImplementedError
