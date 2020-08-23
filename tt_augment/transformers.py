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
        self._img_fwd_transform = None
        self._img_bkd_transform = None

        self._has_reverse = hasattr(self.transformer, "reversal")

    @property
    def window(self):
        return self._window

    @property
    def img_fwd_transform(self):
        return self._img_fwd_transform

    @property
    def img_bkd_transform(self):
        return self._img_bkd_transform

    def apply_fwd_transform(self, image: np.ndarray):
        """
        Apply forward transformation

        :param image:
        :return:
        """
        self._img_fwd_transform = self.transformer(images=image)

    def apply_bkd_transform(self, image: np.ndarray):
        """
        Reverse back the applied transformation

        :param image:
        :return:
        """
        if self._has_reverse:
            self._img_bkd_transform = self.transformer(images=image, do_reversal=True)
        else:
            self._img_bkd_transform = image

    def data(self, image: np.ndarray) -> np.ndarray:
        return image[
            :,
            self.window[0][0] : self.window[0][1],
            self.window[1][0] : self.window[1][1],
            :,
        ]


class TransformerFamily(list):
    """
    Contains the list the Transformers that belong to the same Augmentation class
    """
    def __init__(self, children=None, name=None):
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

        self._inferred_image = None
        self._name = name
        self._child_collation_count = 0

    @property
    def name(self):
        return self._name

    @property
    def inferred_image(self):
        return self._inferred_image

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

        self._inferred_image = np.zeros(image.shape)
        for child in self:
            child.apply_fwd_transform(child.data(image=image))

            yield child

    def add_inferred_to_family(self, image: np.ndarray, child=None, child_index=None):
        assert image.ndim == 4, (
            "Expected image to have shape (batch ,width, height, [channels]), "
            "got shape %s." % (image.shape,)
        )

        if child is None:
            child = self[child_index]
        assert isinstance(
            child, Transformer
        ), "Expected child to be Transformer, got types %s." % (str(type(child)))

        child.apply_bkd_transform(image=image)
        self.dissolve_child(child)
        self._child_collation_count += 1

    def dissolve_child(self, child):
        """
        Merges the output of the child with the family, i.e the previous output

        :param child:
        :return:
        """

        part_1_x = child.window[0][0]
        part_1_y = child.window[0][1]
        part_2_x = child.window[1][0]
        part_2_y = child.window[1][1]

        cropped_image = self._inferred_image[:, part_1_x:part_1_y, part_2_x:part_2_y, :]

        inferred_image = cropped_image + child.img_bkd_transform

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
        self._inferred_image[
            :, part_1_x:part_1_y, part_2_x:part_2_y, :
        ] = inferred_image


@dataclass
class Batch:
    family: TransformerFamily
    child: Transformer
    image: np.ndarray


class TTA:
    def __init__(self, image_dimension: tuple, transformers: list):
        """

        :param image_dimension: (W x H x channels)
        :param transformers: dictionary of transformation to apply
        """
        self.image_dimension = image_dimension
        assert len(self.image_dimension) in [2, 3], (
            "Expected image to have shape (width, height, [channels]), "
            "got shape %s." % (self.image_dimension,)
        )

        self._tt_image = None
        self._tt_family = list()

        self._get_family_collection(transformers)
        self._print_dict = {
            "Transformation": "None",
            "Family": "None",
            "Child": "0/0",
            "Collation Status": "Not Started",
        }

    @property
    def tt_image(self):
        return self._tt_image

    def _get_family_collection(self, transformer):
        for individual_transformer in transformer:
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

            if transformer.transform_dimension > self.image_dimension:
                raise ValueError(
                    "Transformation Dimension Can't be bigger that Image Dimension"
                )
            window = Window.get_window(
                window_size=transformer.transform_dimension,
                org_size=self.image_dimension,
            )

            for win_number, win in window:
                collection.append(Transformer(transformer=transformer, window=win))
            self._tt_family.append(
                TransformerFamily(children=collection, name=transformer_name)
            )

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
        self._tt_image = np.zeros(image.shape)
        for family_iterator, transformer_family in enumerate(self._tt_family):
            self._print_dict["Transformation"] = "{}/{}".format(
                family_iterator + 1, len(self._tt_family)
            )
            self._print_dict["Family"] = transformer_family.name
            for iterator, child in enumerate(transformer_family.get_child(image)):
                self._print_dict["Child"] = "{}/{}".format(
                    iterator + 1, len(transformer_family)
                )
                Printer.print(self._print_dict)

                yield Batch(transformer_family, child, child.img_fwd_transform)

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
            if not np.any(self._tt_image):
                self._tt_image = family.inferred_image
            else:
                self._tt_image += family.inferred_image
                self._tt_image /= 2
