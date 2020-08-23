from dataclasses import dataclass

from tt_augment.printer import Printer

try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
import numpy as np

from tt_augment.tt_custom import custom
from tt_augment.window import Window


class Transformer:
    def __init__(self, transformer, window):
        """

        :param transformer: Augmentation to apply
        :param window: which portion of the image the augmentation to be applied
        """
        self.transformer = transformer

        self._window = window
        self._data_fwd_transform = None
        self._data_bkd_transform = None

        self._has_reverse = hasattr(self.transformer, "reversal")

    @property
    def window(self):
        return self._window

    @property
    def data_fwd_transform(self):
        return self._data_fwd_transform

    @property
    def data_bkd_transform(self):
        return self._data_bkd_transform

    def apply_fwd_transform(self, image: np.ndarray):
        """
        Apply forward transformation

        :param image:
        :return:
        """
        self._data_fwd_transform = self.transformer.fwd(images=image)

    def apply_bkd_transform(self, inferred_data):
        """
        Reverse back the applied transformation

        :param inferred_data:
        :return:
        """
        raise NotImplementedError

    def data(self, image: np.ndarray) -> np.ndarray:
        return image[
            :,
            self.window[0][0] : self.window[0][1],
            self.window[1][0] : self.window[1][1],
            :,
        ]

    def union(self, data):
        raise NotImplementedError


class SegmentationTransformer(Transformer):
    def __init__(self, transformer, window):
        super().__init__(transformer, window)

    def apply_bkd_transform(self, inferred_data: np.ndarray):
        """
        Reverse back the applied transformation

        :param inferred_data:
        :return:
        """
        assert inferred_data.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (inferred_data.shape,)
        )

        if self._has_reverse:
            self._data_bkd_transform = self.transformer.bkd_seg(
                inferred_data=inferred_data
            )
        else:
            self._data_bkd_transform = inferred_data

    def union(self, data: np.ndarray):
        """
        Merges the output of the child with the family, i.e the previous output

        :param data:
        :return:
        """

        part_1_x = self.window[0][0]
        part_1_y = self.window[0][1]
        part_2_x = self.window[1][0]
        part_2_y = self.window[1][1]

        cropped_image = data[:, part_1_x:part_1_y, part_2_x:part_2_y, :]

        inferred_image = cropped_image + self.data_bkd_transform

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
        data[:, part_1_x:part_1_y, part_2_x:part_2_y, :] = inferred_image
        return data


class ClassificationTransformer(Transformer):
    def __init__(self, transformer, window):
        super().__init__(transformer, window)

    def apply_bkd_transform(self, inferred_data: float):
        """
        Reverse back the applied transformation

        :param inferred_data:
        :return:
        """
        if self._has_reverse:
            self._data_bkd_transform = self.transformer.bkd_classification(
                inferred_data=inferred_data
            )
        else:
            self._data_bkd_transform = inferred_data

    def union(self, data: list):
        data.append(self._data_bkd_transform)
        return data


class TransformerFamily(list):
    """
    When the image is split in multiple section, this class hold the sections together as a family, and maintains
    a common output image for all the sections
    """

    def __init__(self, children=None, name=None, family_type=None):
        if children is None:
            list.__init__(self, [])
        elif isinstance(children, Transformer):
            list.__init__(self, [children])
        elif isinstance(children, Iterable):
            assert all([isinstance(child, Transformer) for child in children]), (
                "Expected all children to be Transformer, got types %s."
                % (", ".join([str(type(v)) for v in children]))
            )
            list.__init__(self, children)
        else:
            raise Exception(
                "Expected None or Transformer or list of WindowTransform, "
                "got %s." % (type(children),)
            )
        assert name is not None, "Family Name cant be None"
        assert (
            family_type is not None
        ), "Expected family_type to be [SegmentationTransformer][ClassificationTransformer], got types None"

        self._inferred_data = None

        self._name = name
        self._family_type = family_type
        self._child_collation_count = 0

    @property
    def name(self):
        return self._name

    @property
    def family_type(self):
        return self._family_type

    @property
    def inferred_data(self):
        return self._inferred_data

    @property
    def child_collation_count(self):
        return self._child_collation_count

    def add(self, transform: Transformer):
        self.append(transform)

    def get_child(self, image: np.ndarray) -> Transformer:
        assert image.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (image.shape,)
        )

        if self.family_type == SegmentationTransformer.__name__:
            self._inferred_data = np.zeros(image.shape)
        elif self.family_type == ClassificationTransformer.__name__:
            self._inferred_data = list()
        else:
            raise Exception("Unexpected family type")
        for child in self:
            child.apply_fwd_transform(child.data(image=image))

            yield child

    def add_inferred_to_family(
        self, inferred_data: np.ndarray, child=None, child_index=None
    ):
        if child is None:
            child = self[child_index]
        assert isinstance(
            child, Transformer
        ), "Expected child to be Transformer, got types %s." % (str(type(child)))

        child.apply_bkd_transform(inferred_data=inferred_data)
        self._inferred_data = child.union(self._inferred_data)
        self._child_collation_count += 1

    def transfer_segmentation_inheritance(self, tt_data):
        if not np.any(tt_data):
            tt_data = self.inferred_data
        else:
            tt_data += self.inferred_data
            tt_data /= 2
        return tt_data

    def transfer_inheritance(self, tt_data):
        if self.family_type == SegmentationTransformer.__name__:
            tt_data = self.transfer_segmentation_inheritance(tt_data)
        else:
            raise NotImplementedError
        return tt_data


@dataclass
class Batch:
    family: TransformerFamily
    child: Transformer
    image: np.ndarray


class TTA:
    def __init__(self, image_dimension: tuple, tt_collection: list, tt_type: str):
        """

        :param image_dimension: (W x H x channels)
        :param tt_collection:
        """
        self.image_dimension = image_dimension

        self._tt_data = None
        self._tt_collection = tt_collection
        self._tt_type = tt_type

        self._print_dict = {
            "Transformation": "None",
            "Family": "None",
            "Child": "0/0",
            "Collation Status": "Not Started",
        }

    @property
    def tt_data(self):
        return self._tt_data

    def get_batch(self, image) -> Batch:
        """
        Get batch of transformers on which inference is to be performed

        :param image:
        :return:
        """
        _, w, h, c = image.shape
        assert self.image_dimension == (w, h, c), (
            "Expected image to have shape %s, "
            "got shape %s." % (self.image_dimension, (w, h, c),)
        )

        if self._tt_type == SegmentationTransformer.__name__:
            self._tt_data = np.zeros(image.shape)
        elif self._tt_type == ClassificationTransformer.__name__:
            self._tt_data = list()
        else:
            raise Exception("Unexpected type")
        for family_iterator, transformer_family in enumerate(self._tt_collection):
            self._print_dict["Transformation"] = "{}/{}".format(
                family_iterator + 1, len(self._tt_collection)
            )
            self._print_dict["Family"] = transformer_family.name
            for iterator, child in enumerate(transformer_family.get_child(image)):
                self._print_dict["Child"] = "{}/{}".format(
                    iterator + 1, len(transformer_family)
                )
                Printer.print(self._print_dict)

                yield Batch(transformer_family, child, child.data_fwd_transform)

    def collate_batch(self, image: np.ndarray, batch: Batch):
        """
        Merge the inferred image with the family

        :param image:
        :param batch:
        :return:
        """
        family = batch.family
        child = batch.child

        family.add_inferred_to_family(image, child)
        self._print_dict["Collation Status"] = "{}/{}".format(
            family.child_collation_count + 1, len(family)
        )

        if family.child_collation_count == len(family):
            self._tt_data = family.transfer_inheritance(self.tt_data)

    @classmethod
    def tta_segmentation(cls, image_dimension: tuple, transformers: list):
        assert len(image_dimension) == 3, (
            "Expected image to have shape (width, height, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        seg_collection = list()

        for individual_transformer in transformers:
            assert list(individual_transformer.keys()) == [
                "name",
                "param",
            ], "Expected Keys 'name' and 'param', "
            transformer_name, transformer_param = (
                individual_transformer["name"],
                individual_transformer["param"],
            )

            collection = list()

            transformer = getattr(custom, transformer_name)(**transformer_param)

            if transformer.transform_dimension > image_dimension:
                raise ValueError(
                    "Transformation Dimension Can't be bigger that Image Dimension"
                )
            window = Window.get_window(
                window_size=transformer.transform_dimension, org_size=image_dimension,
            )

            for win_number, win in window:
                collection.append(
                    SegmentationTransformer(transformer=transformer, window=win)
                )
            seg_collection.append(
                TransformerFamily(
                    children=collection,
                    name=transformer_name,
                    family_type=SegmentationTransformer.__name__,
                )
            )
        return cls(
            image_dimension=image_dimension,
            tt_collection=seg_collection,
            tt_type=SegmentationTransformer.__name__,
        )

    @classmethod
    def tta_classification(cls, image_dimension: tuple, transformers: list):
        assert len(image_dimension) == 3, (
            "Expected image to have shape (width, height, [channels]), "
            "got shape %s." % (image_dimension,)
        )

        classification_collection = list()

        for individual_transformer in transformers:
            assert list(individual_transformer.keys()) == [
                "name",
                "param",
            ], "Expected Keys 'name' and 'param', "
            transformer_name, transformer_param = (
                individual_transformer["name"],
                individual_transformer["param"],
            )

            collection = list()

            transformer = getattr(custom, transformer_name)(**transformer_param)

            if transformer.transform_dimension > image_dimension:
                raise ValueError(
                    "Transformation Dimension Can't be bigger that Image Dimension"
                )
            window = Window.get_window(
                window_size=transformer.transform_dimension, org_size=image_dimension,
            )

            for win_number, win in window:
                collection.append(
                    ClassificationTransformer(transformer=transformer, window=win)
                )
            classification_collection.append(
                TransformerFamily(
                    children=collection,
                    name=transformer_name,
                    family_type=ClassificationTransformer.__name__,
                )
            )
        return cls(
            image_dimension=image_dimension,
            tt_collection=classification_collection,
            tt_type=ClassificationTransformer.__name__,
        )
