import numpy as np
from collections import deque

from tt_augment import custom_augmenters


class TransformWindow:
    def __init__(self, stride_size, org_size):
        self.stride_size = stride_size
        self.org_size = org_size
        self.window_collection = self.windows()

    def __len__(self):
        return len(self.window_collection)

    def __getitem__(self, index):
        return index, self.window_collection[index]

    def windows(self):
        """
        W = Columns
        H = Rows
        Input = W x H
        Provides collection of Windows of split_size over img_size, The functions will yield non overlapping
        window if img_size / split_size is divisible, if that's not the case then the function will adjust
        the windows accordingly to accommodate the split_size and yield overlapping windows
        :return:
        """
        if (
            self.stride_size[0] > self.org_size[0]
            or self.stride_size[1] > self.org_size[1]
        ):
            raise ValueError(
                "Size to Split Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(
                    self.stride_size, (self.org_size[0], self.org_size[1])
                )
            )
        cropped_windows = list()

        split_col, split_row = (self.stride_size[0], self.stride_size[1])

        img_col = self.org_size[0]
        img_row = self.org_size[1]

        iter_col = 1
        iter_row = 1

        for col in range(0, img_col, split_col):
            if iter_col == np.ceil(img_col / split_col):
                col = img_col - split_col
            else:
                iter_col += 1
            for row in range(0, img_row, split_row):
                if iter_row == np.ceil(img_row / split_row):
                    row = img_row - split_row
                else:
                    iter_row += 1
                if row + split_row <= img_row and col + split_col <= img_col:
                    cropped_windows.append(
                        ((row, row + split_row), (col, col + split_col))
                    )
            iter_row = 1
        return cropped_windows

    @classmethod
    def get_window(cls, stride_size: tuple, org_size: tuple):
        """
        :param stride_size:
        :param org_size:
        :return:
        """
        return TransformWindow(stride_size, org_size)


class Transform:
    def __init__(self, transformer, window):
        self.transformer = transformer
        self._window = window

    @property
    def window(self):
        return self._window

    def transform(self, image: np.ndarray):
        return self.transformer(images=image)

    def reverse_inferred_transform(self, image: np.ndarray):
        if hasattr(self, "do_reversal"):
            return self.transformer(images=image, do_reversal=True)
        else:
            return image

    def get_window_data(self, image):
        return image[
            :,
            self.window[0][0] : self.window[0][1],
            self.window[1][0] : self.window[1][1],
            :,
        ]

    def add_window_data(self, inferred_image: np.ndarray, tta_image: np.ndarray):
        """

        :param inferred_image:
        :param tta_image:
        :return:
        """

        part_1_x = self.window[0][0]
        part_1_y = self.window[0][1]
        part_2_x = self.window[1][0]
        part_2_y = self.window[1][1]

        cropped_image = tta_image[:, part_1_x:part_1_y, part_2_x:part_2_y, :]

        inferred_image = cropped_image + inferred_image

        if np.any(cropped_image):
            intersecting_prediction_elements = np.zeros(cropped_image.shape)
            intersecting_prediction_elements[cropped_image > 0] = 1

            non_intersecting_prediction_elements = 1 - intersecting_prediction_elements

            intersected_prediction = inferred_image * intersecting_prediction_elements
            aggregate_prediction = intersected_prediction / 2

            non_intersected_prediction = np.multiply(
                non_intersecting_prediction_elements, inferred_image
            )
            inferred_image = aggregate_prediction + non_intersected_prediction
        tta_image[:, part_1_x:part_1_y, part_2_x:part_2_y, :] = inferred_image
        return tta_image


class TTA:
    def __init__(self, network_input_dimension: tuple, transformers: dict):
        self.network_input_dimension = network_input_dimension
        self.collection = self.make_transformer(transformers)

        self.collate_transform = deque()

        self.tta_image = None

    def make_transformer(self, transformers):
        collection = list()
        for individual_transformer, transformer_param in transformers.items():
            transformation = getattr(custom_augmenters, individual_transformer)(
                **transformer_param
            )
            window = TransformWindow.get_window(
                transformation.dimension, self.network_input_dimension
            )

            for win_number, win in window:
                collection.append(Transform(transformer=transformation, window=win))
        return collection

    def tta_gen(self, image: np.ndarray):
        self.tta_image = np.zeros(image.shape)

        for transformation in self.collection:
            yield transformation, transformation.get_window_data(image)

    def collate_inference(self, transformation: Transform, image: np.ndarray):
        self.tta_image = transformation.add_window_data(image, self.tta_image)
        # return transformation.reverse_inferred_transform(image)
