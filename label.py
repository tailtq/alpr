import numpy as np


class Label:
    def __init__(self, tl, tr, br, bl):
        self.coordinates = {
            "tl": np.array(tl),
            "tr": np.array(tr),
            "br": np.array(br),
            "bl": np.array(bl),
        }

    def tl(self): return self.coordinates["tl"]

    def tr(self): return self.coordinates["tr"]

    def br(self): return self.coordinates["br"]

    def bl(self): return self.coordinates["bl"]

    def cc(self): return self.coordinates["tl"] + self.wh() / 2

    def wh(self): return np.abs(self.coordinates["br"] - self.coordinates["tl"])

    @staticmethod
    def read_n_parse_label(link, should_convert=False, img_shape=[0, 0]):
        """
        Read and convert image to regular coordinates
        :param link:
        :param should_convert:
        :param img_shape:
        :return:
        """
        line = open(link, "r").readlines()[0].split(",")
        coordinates = [float(e) for e in line[1:9]]
        coordinates_x = coordinates[:4]
        coordinates_y = coordinates[4:]
        height, width = img_shape[:2]

        min_index = coordinates.index(min(coordinates[:4]))

        if should_convert:
            return Label(
                [int(coordinates_x[min_index % 4] * width), int(coordinates_y[min_index % 4] * height)],
                [int(coordinates_x[(min_index + 1) % 4] * width), int(coordinates_y[(min_index + 1) % 4] * height)],
                [int(coordinates_x[(min_index + 2) % 4] * width), int(coordinates_y[(min_index + 2) % 4] * height)],
                [int(coordinates_x[(min_index + 3) % 4] * width), int(coordinates_y[(min_index + 3) % 4] * height)],
            )

        return Label(
            [coordinates_x[min_index % 4], coordinates_y[min_index % 4]],
            [coordinates_x[(min_index + 1) % 4], coordinates_y[(min_index + 1) % 4]],
            [coordinates_x[(min_index + 2) % 4], coordinates_y[(min_index + 2) % 4]],
            [coordinates_x[(min_index + 3) % 4], coordinates_y[(min_index + 3) % 4]],
        )
