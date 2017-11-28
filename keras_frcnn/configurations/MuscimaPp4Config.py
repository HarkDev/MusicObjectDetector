import math

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class MuscimaPp4Config(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(anchor_box_scales=[1, 2, 4, 8, 16, 24],
                         anchor_box_ratios=[[1, 1],
                                            [1 / math.sqrt(4), 4 / math.sqrt(4)],
                                            [2 / math.sqrt(2), 1 / math.sqrt(2)]],
                         resize_smallest_side_of_image_to=300,
                         number_of_ROIs_at_once=128)

    def name(self) -> str:
        return "muscima_pp_4"


if __name__ == "__main__":
    configuration = MuscimaPp4Config()
    print(configuration.summary())
