import shutil
import unittest
from typing import List, Tuple

import os

from keras_frcnn.muscima_image_cutter import create_annotations_in_plain_format, create_annotations_in_pascal_voc_format


class MuscimaImageCutterTest(unittest.TestCase):
    def test_create_annotations_in_plain_format(self):
        # Arrange
        objects_appearing_in_cropped_image = self.get_fake_annotations()
        annotation_file = "test_annotations.txt"
        self.delete_annotations_if_exist(annotation_file)

        # Act
        create_annotations_in_plain_format(annotation_file, objects_appearing_in_cropped_image)

        # Assert
        self.assertTrue(os.path.exists(annotation_file))
        with open(annotation_file, "r") as file:
            first_line = file.readline()
            self.assertEqual("file1,5,5,10,10,class1\n", first_line)

    def test_create_annotations_in_pascal_voc_format_expect_files_to_be_generated(self):
        # Arrange
        objects_appearing_in_cropped_image = self.get_fake_annotations()
        annotations_folder = "Test-Annotations"
        shutil.rmtree(annotations_folder, ignore_errors=True)

        # Act
        create_annotations_in_pascal_voc_format(annotations_folder, objects_appearing_in_cropped_image)

        # Assert
        self.assertTrue(os.path.exists(annotations_folder))
        number_of_generated_annotation_files = len(os.listdir(annotations_folder))
        self.assertEqual(number_of_generated_annotation_files, 2, "Expecting 1 file per image file")


    def get_fake_annotations(self):
        objects_appearing_in_cropped_image: List[Tuple[str, str, Tuple[int, int, int, int]]] = []
        objects_appearing_in_cropped_image.append(("file1", "class1", [5, 5, 10, 10]))
        objects_appearing_in_cropped_image.append(("file2", "class1", [12, 13, 14, 16]))
        objects_appearing_in_cropped_image.append(("file2", "class2", [25, 25, 20, 20]))
        return objects_appearing_in_cropped_image

    def delete_annotations_if_exist(self, annotation_file):
        if (os.path.exists(annotation_file)):
            os.remove(annotation_file)


if __name__ == '__main__':
    unittest.main()
