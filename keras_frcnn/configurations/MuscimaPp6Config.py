import math

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration


class MuscimaPp6Config(FasterRcnnConfiguration):
    def __init__(self):
        super().__init__(anchor_box_scales=[4, 8, 16, 24],
                         anchor_box_ratios=[[1, 1],
                                            [1 / math.sqrt(4), 4 / math.sqrt(4)],
                                            [2 / math.sqrt(2), 1 / math.sqrt(2)]],
                         scale_images=False,
                         rpn_stride=8,
                         number_of_ROIs_at_once=200,
                         )

    def name(self) -> str:
        return "muscima_pp_6"


if __name__ == "__main__":
    configuration = MuscimaPp6Config()
    print(configuration.summary())
