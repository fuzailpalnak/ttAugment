import numpy as np


class Window:
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
        return Window(window_size, org_size)
