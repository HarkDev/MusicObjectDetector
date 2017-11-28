import math

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class MuscimaPp3Config(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(anchor_box_scales=[1, 2, 4, 8, 16],
                         anchor_box_ratios=[[1, 1],
                                            [4 / math.sqrt(4), 1 / math.sqrt(4)],
                                            [1 / math.sqrt(2), 2 / math.sqrt(2)]],
                         resize_smallest_side_of_image_to=300,
                         number_of_ROIs_at_once=64)

    def name(self) -> str:
        return "muscima_pp_3"


if __name__ == "__main__":
    configuration = MuscimaPp3Config()
    print(configuration.summary())
