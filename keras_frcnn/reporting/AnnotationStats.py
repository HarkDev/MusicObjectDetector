import argparse
import numpy as np
from collections import Counter
from typing import List


class Box(object):

    """Docstring for Box. """

    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """TODO: to be defined1.

        :x1: TODO
        :y1: TODO
        :x2: TODO
        :y2: TODO

        """
        self._x1 = x1
        self._y1 = y1
        self._x2 = x2
        self._y2 = y2

    @property
    def width(self) -> float:
        return self._x2 - self._x1

    @property
    def height(self) -> float:
        return self._y2 - self._y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def normalize(self, other):
        return Box(self._x1 / other.width,
                   self._y1 / other.height,
                   self._x2 / other.width,
                   self._y2 / other.height)


def stat(values: np.array) -> List:
    min_values = min(values)
    max_values = max(values)
    mean_values = np.mean(values)
    std_values = np.std(values)
    return [min_values, max_values, mean_values, std_values]


class AnnotationStats(object):

    """AnnotationStats computes statistics on annotation file"""

    def __init__(self, annotation_file: str, crop_bounding_boxes) -> None:
        """Constructor"""
        self.annotation_file = annotation_file
        self.crop_bounding_boxes = crop_bounding_boxes

    def compute(self):
        data = np.genfromtxt(
            self.annotation_file,
            dtype=None,
            delimiter=',')
        # compute stats on nbr of objects
        filenames = [x[0].decode() for x in data]
        nbr_object_stats = Counter(filenames)
        values = list(nbr_object_stats.values())
        stat_values = stat(values)
        print("------------------------------------------------------------")
        print("Statistics on number of objects in cropped images")
        print("min:", stat_values[0])
        print("max:", stat_values[1])
        print("mean:", stat_values[2])
        print("std:", stat_values[3])

        # compute stats on areas and width height ratio of objects
        # object coord are normalized by the size of the input image
        boxes = [Box(x[1], x[2], x[3], x[4]) for x in data]
        images_sizes = np.load(self.crop_bounding_boxes)['crop_bounding_boxes'][()]
        for i, filename in enumerate(filenames):
            image_box = Box(*images_sizes[filename][0])
            boxes[i] = boxes[i].normalize(image_box)
        widths = np.array([x.width for x in boxes])
        heights = np.array([x.height for x in boxes])
        areas = np.array([x.area for x in boxes])
        ratios = widths / heights
        stat_areas = stat(areas)
        stat_ratios = stat(ratios)
        print("------------------------------------------------------------")
        print("Statistics on areas and w/h ratio of objects in cropped images")
        print("area, w/h min:", stat_areas[0], stat_ratios[0])
        print("area, w/h max:", stat_areas[1], stat_ratios[1])
        print("area, w/h mean:", stat_areas[2], stat_ratios[2])
        print("area, w/h std:", stat_areas[3], stat_ratios[3])

        print("------------------------------------------------------------")
        print("Statistics using typical input images of 600x300")
        stat_areas = np.sqrt(np.array(stat_areas) * 600 * 300)
        print("square box size min:", stat_areas[0])
        print("square box size max:", stat_areas[1])
        print("square box size mean:", stat_areas[2])
        print("square box size std:", stat_areas[3])

if __name__ == "__main__":
    """
    $ # To be run using something like this
    $ python keras_frcnn/reporting/AnnotationStats.py data/muscima_pp_cropped_images/Annotations.txt data/muscima_pp_cropped_images/crop_bounding_boxes.npz
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("annotation_path", type=str,
                        help="Path to annotation file")
    parser.add_argument("crop_bounding_boxes", type=str,
                        help="Path to cropped images sizes npz file")
    args = parser.parse_args()
    stats = AnnotationStats(args.annotation_path,
                            args.crop_bounding_boxes)
    stats.compute()
