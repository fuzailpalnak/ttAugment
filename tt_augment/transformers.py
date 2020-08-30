from tt_augment import tt_custom, arithmetic_aggregate
from tt_augment.printer import Printer

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom, TTCustom
from tt_augment.window import Window


class TTInfo:
    def __init__(self, transformer, window):

        self._transformer = transformer
        self._window = window
        self._data = None
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

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def get_windowed_image(self, image: np.ndarray) -> np.ndarray:
        return image[
            :,
            self.window[0][0] : self.window[0][1],
            self.window[1][0] : self.window[1][1],
            :,
        ]


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

        part_1_x = tt_info.window[0][0]
        part_1_y = tt_info.window[0][1]
        part_2_x = tt_info.window[1][0]
        part_2_y = tt_info.window[1][1]

        cropped_image = self.tt_member_data[:, part_1_x:part_1_y, part_2_x:part_2_y, :]

        inferred_image = cropped_image + tt_info.data

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
        self.tt_member_data[:, part_1_x:part_1_y, part_2_x:part_2_y, :] = inferred_image


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
    assert (
        len(image_dimension) == 3
    ), "Expected image to have shape (width, height, [channels]), " "got shape %s." % (
        image_dimension,
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
            window_size=transformer.transform_dimension,
            org_size=image_dimension,
        )

        for win_number, win in window:
            member.add_image_info(TTInfo(transformer=transformer, window=win))
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


def segmentation_binary(
    image_dimension: tuple,
    batch_size: int,
    transformers: list,
    arithmetic_compute="mean",
):
    assert (
        len(image_dimension) == 3
    ), "Expected image to have shape (width, height, [channels]), " "got shape %s." % (
        image_dimension,
    )

    assert arithmetic_compute in [
        "mean",
        "geometric_mean",
    ], "Expected values ['mean', 'geometric_mean'], " "got %s." % (arithmetic_compute,)
    w, h, _ = image_dimension

    if arithmetic_compute == "mean":
        inference_data = np.zeros((batch_size, w, h, 1))
    else:
        inference_data = np.ones((batch_size, w, h, 1))
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
    batch_size: int,
    transformers: list,
    arithmetic_compute="mean",
):
    assert (
        len(image_dimension) == 3
    ), "Expected image to have shape (width, height, [channels]), " "got shape %s." % (
        image_dimension,
    )

    assert arithmetic_compute in [
        "mean",
        "geometric_mean",
    ], "Expected values ['mean', 'geometric_mean'], " "got %s." % (arithmetic_compute,)
    w, h, _ = image_dimension

    if arithmetic_compute == "mean":
        inference_data = np.zeros((batch_size, w, h, 3))
    else:
        inference_data = np.ones((batch_size, w, h, 3))
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


from timeit import default_timer as timer

import tifffile as tiff

a = tiff.imread(r"D:\Cypherics\Library\test\10528705_15.tiff")
b = tiff.imread(r"D:\Cypherics\Library\test\10078735_15.tiff")

d = [a, b]
# for i in range(32):
#     d.append(a)
d = np.array(d)
import time

now = time.time()

tta = segmentation_color(
    image_dimension=(1500, 1500, 3),
    batch_size=2,
    transformers=[
        {
            "name": "Rot",
            "param": {"angle": 90},
            "transform_dimension": (384, 384, 3),
            "network_dimension": (384, 384, 3),
        },
        {
            "name": "GaussianBlur",
            "param": {},
            "transform_dimension": (384, 384, 3),
            "network_dimension": (384, 384, 3),
        },
    ],
)
end_time = time.time()

print("total time taken by Initalization: ", end_time - now)
import cv2

# images = np.array(
#     [ia.quokka(size=(512, 512)) for _ in range(2)],
#     dtype=np.uint8
# )
# d1 = Fliplr(1)
# f = d1(images=images)
import time

now = time.time()

for tt_info1 in tta.tt(d):
    ff_image = tta.tt_fwd(tt_info1)
    tta.add_inference(tta.tt_bkd(ff_image, tt_info1))
end_time = time.time()

print("total time taken by collate batch: ", end_time - now)
for j in range(2):
    cv2.imwrite("g_{}.png".format(j), tta.tt_family_data[j])
