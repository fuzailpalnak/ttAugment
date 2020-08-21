import numpy as np

from tt_augment.tt_custom import tt_fwd_bkd


class TransformWindow:
    def __init__(self, window_size, org_size):
        self.window_size = window_size
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
            self.window_size[0] > self.org_size[0]
            or self.window_size[1] > self.org_size[1]
        ):
            raise ValueError(
                "Window Size Can't Be Greater than Image, Given {},"
                " Expected <= {}".format(
                    self.window_size, (self.org_size[0], self.org_size[1])
                )
            )
        cropped_windows = list()

        split_col, split_row = (self.window_size[0], self.window_size[1])

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
    def get_window(cls, window_size: tuple, org_size: tuple):
        """
        :param window_size:
        :param org_size:
        :return:
        """
        return TransformWindow(window_size, org_size)


class Transform:
    def __init__(self, transformer, window):
        self.transformer = transformer
        self._window = window

        self._has_reverse = hasattr(self.transformer, "reversal")

    @property
    def window(self):
        return self._window

    def transform(self, image: np.ndarray, path: str):
        if path == "fwd":
            return self.transformer(images=image)
        elif path == "bkd":
            if self._has_reverse:
                return self.transformer(images=image, do_reversal=True)
            else:
                return image
        else:
            raise NotImplementedError

    def data(self, image):
        return image[
            :,
            self.window[0][0] : self.window[0][1],
            self.window[1][0] : self.window[1][1],
            :,
        ]


class TTA:
    def __init__(self, image_dimension: tuple, transformers: dict):
        self.image_dimension = image_dimension
        self.transformers = transformers

        self.collection = self.make_transformer()

        self._tt_image = None

    @property
    def tt_image(self):
        return self._tt_image

    def make_transformer(self):
        collection = list()
        for individual_transformer, transformer_param in self.transformers.items():
            transformation = getattr(tt_fwd_bkd, individual_transformer)(
                **transformer_param
            )
            window = TransformWindow.get_window(
                window_size=transformation.dimension, org_size=self.image_dimension,
            )

            for win_number, win in window:
                collection.append(Transform(transformer=transformation, window=win))
        return collection

    def tta_gen(self, image: np.ndarray):
        self._tt_image = np.zeros(image.shape)

        for transformation in self.collection:
            yield transformation, transformation.transform(
                transformation.data(image=image), path="fwd"
            )

    def aggregate(self, inferred_image: np.ndarray, window: tuple):
        """

        :param window:
        :param inferred_image:
        :return:
        """

        part_1_x = window[0][0]
        part_1_y = window[0][1]
        part_2_x = window[1][0]
        part_2_y = window[1][1]

        cropped_image = self._tt_image[:, part_1_x:part_1_y, part_2_x:part_2_y, :]

        inferred_image = cropped_image + inferred_image

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
        self._tt_image[:, part_1_x:part_1_y, part_2_x:part_2_y, :] = inferred_image

    def collate_inference(self, transformation: Transform, inferred_image: np.ndarray):
        inferred_image = transformation.transform(image=inferred_image, path="bkd")
        self.aggregate(inferred_image=inferred_image, window=transformation.window)
