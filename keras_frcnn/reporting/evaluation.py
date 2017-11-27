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

import numpy as np
import argparse
from typing import Dict
from sklearn import preprocessing
from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluation
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ground_truth", type=str,
                        help="Path to ground_truth file")
    parser.add_argument("detection", type=str,
                        help="Path to detection results file")
    args = parser.parse_args()

    gt = np.genfromtxt(
        args.ground_truth, dtype=None, delimiter=',')
    class_labels = list(set([x[5].decode() for x in gt]))
    le = preprocessing.LabelEncoder()
    le.fit(class_labels)
    num_groundtruth_classes = len(class_labels)
    print(num_groundtruth_classes)
    evaluator = ObjectDetectionEvaluation(
        num_groundtruth_classes)

    gt_dic: Dict = {}
    for (image_key, xmin, ymin, xmax, ymax, class_label) in tqdm(gt):
        if image_key not in gt_dic:
            gt_dic[image_key] = {}
            gt_dic[image_key]["boxes"] = []
            gt_dic[image_key]["class_labels"] = []
        gt_dic[image_key]["boxes"].append([ymin, xmin, ymax, xmax])
        gt_dic[image_key]["class_labels"].append(class_label)
    for key, value in tqdm(gt_dic.items()):
        evaluator.add_single_ground_truth_image_info(
            key, np.array(value["boxes"], dtype="float32"),
            le.transform(value["class_labels"]))

    detect = np.genfromtxt(
        args.detection, dtype=None, delimiter=',')
    detect_dic: Dict = {}
    for (image_key, xmin, ymin, xmax, ymax, class_label, score) in tqdm(detect):
        if image_key not in detect_dic:
            detect_dic[image_key] = {}
            detect_dic[image_key]["boxes"] = []
            detect_dic[image_key]["class_labels"] = []
            detect_dic[image_key]["scores"] = []
        detect_dic[image_key]["boxes"].append([ymin, xmin, ymax, xmax])
        detect_dic[image_key]["class_labels"].append(class_label)
        detect_dic[image_key]["scores"].append(score)
    for key, value in tqdm(detect_dic.items()):
        evaluator.add_single_detected_image_info(
            key, np.array(value["boxes"], dtype="float32"),
            np.array(value["scores"]), le.transform(value["class_labels"]))
    AP_per_class, mAP, precisions_per_class, recalls_per_class, \
        corloc_per_class, mean_corloc = evaluator.evaluate()
    print("mAP:", mAP)
    # print("mean, std precision:", np.mean(precisions_per_class),
          # np.std(precisions_per_class))
    # print("mean, std recall:", np.mean(precisions_per_class),
          # np.std(precisions_per_class))
    print("mean corlocs:", mean_corloc)
