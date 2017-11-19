from typing import List

from keras_frcnn.configurations.FasterRcnnConfiguration import FasterRcnnConfiguration
from keras_frcnn.configurations.ManyAnchorBoxRatiosConfig import ManyAnchorBoxRatiosConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesManyRoisLargeStrideConfig import \
    ManyAnchorBoxScalesManyRoisLargeStrideConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesManyRoisMediumStrideConfig import \
    ManyAnchorBoxScalesManyRoisMediumStrideConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesManyRoisSmallStrideConfig import \
    ManyAnchorBoxScalesManyRoisSmallStrideConfig
from keras_frcnn.configurations.SmallAnchorBoxScalesManyRoisConfig import SmallAnchorBoxScalesManyRoisConfig
from keras_frcnn.configurations.SmallImagesConfig import SmallImagesConfig
from keras_frcnn.configurations.StretchedAnchorBoxRatiosConfig import StretchedAnchorBoxRatiosConfig
from keras_frcnn.configurations.ManyAnchorBoxScalesConfig import ManyAnchorBoxScalesConfig
from keras_frcnn.configurations.OriginalPascalVocConfig import OriginalPascalVocConfig
from keras_frcnn.configurations.SmallAnchorBoxScalesConfig import SmallAnchorBoxScalesConfig
from keras_frcnn.configurations.AutomaticConfig import AutomaticConfig
from keras_frcnn.networks.FasterRcnnNetwork import FasterRcnnNetwork
from keras_frcnn.networks.ResNet50 import ResNet50
from keras_frcnn.networks.SimpleResNet import SimpleResNet
from keras_frcnn.networks.SimpleVgg import SimpleVgg
from keras_frcnn.networks.Vgg16 import Vgg16


class NetworkFactory:
    @staticmethod
    def get_network_by_name(name: str) -> FasterRcnnNetwork:

        configurations = NetworkFactory.get_all_configurations()

        for i in range(len(configurations)):
            if configurations[i].name() == name:
                return configurations[i]

        raise Exception("No configuration found by name {0}".format(name))

    @staticmethod
    def get_all_configurations() -> List[FasterRcnnNetwork]:
        configurations = [ResNet50(),
                          Vgg16(),
                          SimpleResNet(),
                          SimpleVgg()]
        return configurations


if __name__ == "__main__":
    configurations = NetworkFactory.get_all_configurations()
    print("Available networks are:")
    for configuration in configurations:
        print("- " + configuration.name())
