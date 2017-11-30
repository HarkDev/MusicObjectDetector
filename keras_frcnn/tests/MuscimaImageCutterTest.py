import unittest
from typing import List, Tuple

import os

from keras_frcnn.muscima_image_cutter import create_annotations_in_plain_format

class MuscimaImageCutterTest(unittest.TestCase):
    def test_create_annotations_in_plain_format(self):
        # Arrange
        objects_appearing_in_cropped_image: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
        objects_appearing_in_cropped_image.append(("file1", "class1", [5,5,10,10]))
        objects_appearing_in_cropped_image.append(("file2", "class1", [12,13,14,16]))
        objects_appearing_in_cropped_image.append(("file2", "class2", [25,25,20,20]))
        annotation_file = "test_annotations.txt"
        self.delete_annotations_if_exist(annotation_file)

        # Act
        create_annotations_in_plain_format(annotation_file, objects_appearing_in_cropped_image)

        # Assert
        self.assertTrue(os.path.exists(annotation_file))
        with open(annotation_file, "r") as file:
            first_line = file.readline()
            self.assertEqual("file1,5,5,10,10,class1\n", first_line)

    def delete_annotations_if_exist(self, annotation_file):
        if (os.path.exists(annotation_file)):
            os.remove(annotation_file)


if __name__ == '__main__':
    unittest.main()
