import math
from keras import backend as K

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class MuscimaPp1Config(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(anchor_box_scales=[1, 2, 4, 8, 16],
                         anchor_box_ratios=[[1, 1],
                                            [1 / math.sqrt(4), 4 / math.sqrt(4)],
                                            [2 / math.sqrt(2), 1 / math.sqrt(2)]],
                         resize_smallest_side_of_image_to=300)

    def name(self) -> str:
        return "muscima_pp_1"


if __name__ == "__main__":
    configuration = MuscimaPp1Config()
    print(configuration.summary())
