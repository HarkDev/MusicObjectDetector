"""object_detection_evaluation module.

ObjectDetectionEvaluation is a class which manages ground truth information of a
object detection dataset, and computes frequently used detection metrics such as
Precision, Recall, CorLoc of the provided detection results.
It supports the following operations:
1) Add ground truth information of images sequentially.
2) Add detection result of images sequentially.
3) Evaluate detection metrics on already inserted detection results.
4) Write evaluation result into a pickle file for future processing or
   visualization.

Note: This module operates on numpy boxes and box lists.
"""

import os
import numpy as np
import argparse
from typing import Dict
from sklearn import preprocessing
from object_detection.utils.object_detection_evaluation import \
    ObjectDetectionEvaluation
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gt", "--ground_truth_annotations_file_path", type=str,
                        dest="ground_truth_annotations_file_path",
                        help="Path to file that contains annotations for the ground truth detections")
    parser.add_argument("-detect", "--detected_bounding_boxes_file_path", type=str,
                        dest="detected_bounding_boxes_file_path",
                        help="Path where the detection results reside")

    options, unparsed = parser.parse_known_args()

    if not options.ground_truth_annotations_file_path:
        raise ValueError('Error: Must provide a valid file path to the annotations file containing the ground-truth. '
                         'Pass --ground_truth_annotations_file_path <FileName> to command line')
    if not options.detected_bounding_boxes_file_path:
        raise ValueError('Error: Must provide a valid file path to the file containing the detected bounding boxes. '
                         'Pass --detected_bounding_boxes_file_path <FileName> to command line')

    ground_truth_annotations_file_path = options.ground_truth_annotations_file_path

    for detected_bounding_boxes_file_path in tqdm(os.listdir(options.detected_bounding_boxes_file_path),
                                                  desc="Computing statistics for results"):

        detected_bounding_boxes_file_path = os.path.join(options.detected_bounding_boxes_file_path,
                                                         detected_bounding_boxes_file_path)

        # load ground-truth and detections
        ground_truth_file_reader = np.genfromtxt(ground_truth_annotations_file_path, dtype=None, delimiter=',')
        detection_results_file_reader = np.genfromtxt(detected_bounding_boxes_file_path, dtype=None, delimiter=',')

        # process class labels
        class_labels = list(set([x[5].decode() for x in ground_truth_file_reader]))
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(class_labels)

        number_of_ground_truth_classes = len(class_labels)
        print("Number of ground truth classes: {0}".format(number_of_ground_truth_classes))

        # filter out non used gt files
        file_names_in_ground_truth = np.array([x[0] for x in ground_truth_file_reader])
        file_names_in_detection_results = np.array(list(set([x[0] for x in detection_results_file_reader])))

        indexes = np.array([], dtype=int)
        not_found = []
        for f in file_names_in_detection_results:
            if len(np.where(file_names_in_ground_truth == f)[0]) == 0:
                print(f, "not found")
                not_found.append(f)
            indexes = np.append(indexes, np.where(file_names_in_ground_truth == f)[0])
        ground_truth_file_reader = ground_truth_file_reader[indexes]
        file_names_in_ground_truth = set([x[0] for x in ground_truth_file_reader] + not_found)
        file_names_in_detection_results = set([x[0] for x in detection_results_file_reader])
        print(len(file_names_in_ground_truth), len(file_names_in_detection_results))
        assert (set(file_names_in_ground_truth) == set(file_names_in_detection_results))

        evaluator = ObjectDetectionEvaluation(
            number_of_ground_truth_classes)
        gt_dic: Dict = {}
        for (image_key, x_min, y_min, x_max, y_max, class_label) in ground_truth_file_reader:
            if image_key not in gt_dic:
                gt_dic[image_key] = {}
                gt_dic[image_key]["boxes"] = []
                gt_dic[image_key]["class_labels"] = []
            gt_dic[image_key]["boxes"].append([y_min, x_min, y_max, x_max])
            gt_dic[image_key]["class_labels"].append(class_label)
        for key, value in gt_dic.items():
            evaluator.add_single_ground_truth_image_info(
                key, np.array(value["boxes"], dtype="float32"),
                label_encoder.transform(value["class_labels"]))

        detect_dic: Dict = {}
        for (image_key, x_min, y_min, x_max, y_max, class_label, score) in detection_results_file_reader:
            if image_key not in detect_dic:
                detect_dic[image_key] = {}
                detect_dic[image_key]["boxes"] = []
                detect_dic[image_key]["class_labels"] = []
                detect_dic[image_key]["scores"] = []
            detect_dic[image_key]["boxes"].append([y_min, x_min, y_max, x_max])
            detect_dic[image_key]["class_labels"].append(class_label)
            detect_dic[image_key]["scores"].append(score / 100.)
        for key, value in detect_dic.items():
            evaluator.add_single_detected_image_info(
                key, np.array(value["boxes"], dtype="float32"),
                np.array(value["scores"]), label_encoder.transform(value["class_labels"]))
        average_precision_per_class, mean_average_precision, precisions_per_class, recalls_per_class, \
        corloc_per_class, mean_corloc = evaluator.evaluate()
        print("--- {0} ---".format(os.path.basename(detected_bounding_boxes_file_path)))
        # print(average_precision_per_class)
        print("mAP:", mean_average_precision)
        # print("mean, std precision:", np.mean(precisions_per_class), np.std(precisions_per_class))
        # print("mean, std recall:", np.mean(precisions_per_class), np.std(precisions_per_class))
        print("mean corlocs:", mean_corloc)
        filename = detected_bounding_boxes_file_path.replace("Results", "Metrics")
        if not os.path.exists(os.path.split(filename)[0]):
            os.makedirs(os.path.split(filename)[0])
        f = open(filename, "w")
        f.write(detected_bounding_boxes_file_path + "\n")
        f.write("mAP: " + str(mean_average_precision) + "\n")
        f.write("mean corlocs: " + str(mean_corloc) + "\n")
        f.close()
