# Music Object Detector

This repository is the home of a Faster R-CNN implementation for Music Symbols to implement a fast and reliable Music Symbol detector with Deep Learning.

[![Build Status](https://travis-ci.org/apacha/MusicObjectDetector.svg?branch=master)](https://travis-ci.org/apacha/MusicObjectDetector)
[![codecov](https://codecov.io/gh/apacha/MusicObjectDetector/branch/master/graph/badge.svg)](https://codecov.io/gh/apacha/MusicObjectDetector)
[![Code Health](https://landscape.io/github/apacha/MusicObjectDetector/master/landscape.svg?style=flat)](https://landscape.io/github/apacha/MusicObjectDetector/master)

Note my previous projects that [classified entire sheets](https://github.com/apacha/MusicScoreClassifier) or [learnt to classify different music symbols](https://github.com/apacha/MusicSymbolClassifier).

An extensive overview of the results of different parameters is documented in this [Google Spreadsheet](https://docs.google.com/spreadsheets/d/1MT4CH9yJD_vM9nT8JgnfmzwAVIuRoQYEyv-5FHMjYVo/edit?usp=sharing).

# Running the application
This repository contains several scripts that can be used independently of each other. 
Before running them, make sure that you have the necessary requirements installed. 

## Requirements

- Python 3.6
- Keras 2.0.9
- Tensorflow 1.4.0 (or optionally tensorflow-gpu 1.4.0)
- [Microsoft Visual C++ Build Tools 2015](http://landinghub.visualstudio.com/visual-cpp-build-tools) (for faster data_generator)

Optional: If you want to print the graph of the model being trained, install GraphViz on Windows via http://www.graphviz.org/Download_windows.php and add /bin to the PATH or run `sudo apt-get install graphviz` on Ubuntu (see https://github.com/fchollet/keras/issues/3210)

For installing Tensorflow and Keras we recommend using [Anaconda](https://www.continuum.io/downloads) or 
[Miniconda](https://conda.io/miniconda.html) as Python distribution (we did so for preparing Travis-CI and it worked).

To accelerate training even further, you can make use of your GPU, by installing tensorflow-gpu instead of tensorflow
via pip (note that you can only have one of them) and the required Nvidia drivers. For Windows, we recommend the
[excellent tutorial by Phil Ferriere](https://github.com/philferriere/dlwin). For Linux, we recommend using the
 official tutorials by [Tensorflow](https://www.tensorflow.org/install/) and [Keras](https://keras.io/#installation).

## Training the model

The easiest way to start the training is to run `TrainModel.ps` from the PowerShell.

### Manually start the training
For manually starting the training, make sure to first compile the tools 

```commandline
cd keras_frcnn/py_faster_rcnn
python setup.py build_ext --inplace
```

then run TrainModel like this

    MusicObjectDetector> python TrainModel.py --network resnet50 --output_weight_path "resnet50.hdf5"

## Evaluate results

Since we are using the evaluation tools from the Google Object Detection API, we need to install a few things first (as indicated by the requirements.txt file):

### Linux
See the travis.yml file for an automatic way of installing the required dependencies.
Basically you need to make sure you have [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) installed first to be able to run `protoc`.

Clone https://github.com/tensorflow/models, e.g. 

`git clone https://github.com/tensorflow/models tensorflow-models`

Then build the required libraries

```commandline
cd tensorflow-models/research
protoc object_detection/protos/*.proto --python_out=.
cd slim
python setup.py install
cd ..
python setup.py install
```

### Windows
First, make sure you have [protocol buffers](https://developers.google.com/protocol-buffers/docs/downloads) installed, by heading over to [the download page](https://github.com/google/protobuf/releases/tag/v2.6.0) and download the version 2.6.0. Extract and copy the protoc.exe to a place, where you can run it from later on.  

Clone https://github.com/tensorflow/models, e.g. 

`git clone https://github.com/tensorflow/models tensorflow-models`

```commandline
cd tensorflow-models\research
protoc object_detection/protos/*.proto --python_out=.
```
if protoc does not understand the *-operator, build the files individually:
```commandline
protoc object_detection\protos\anchor_generator.proto               --python_out=.
protoc object_detection\protos\argmax_matcher.proto                 --python_out=.
protoc object_detection\protos\bipartite_matcher.proto              --python_out=.
protoc object_detection\protos\box_coder.proto                      --python_out=.
protoc object_detection\protos\box_predictor.proto                  --python_out=.
protoc object_detection\protos\eval.proto                           --python_out=.
protoc object_detection\protos\faster_rcnn.proto                    --python_out=.
protoc object_detection\protos\faster_rcnn_box_coder.proto          --python_out=.
protoc object_detection\protos\grid_anchor_generator.proto          --python_out=.
protoc object_detection\protos\hyperparams.proto                    --python_out=.
protoc object_detection\protos\image_resizer.proto                  --python_out=.
protoc object_detection\protos\input_reader.proto                   --python_out=.
protoc object_detection\protos\keypoint_box_coder.proto             --python_out=.
protoc object_detection\protos\losses.proto                         --python_out=.
protoc object_detection\protos\matcher.proto                        --python_out=.
protoc object_detection\protos\mean_stddev_box_coder.proto          --python_out=.
protoc object_detection\protos\model.proto                          --python_out=.
protoc object_detection\protos\optimizer.proto                      --python_out=.
protoc object_detection\protos\pipeline.proto                       --python_out=.
protoc object_detection\protos\post_processing.proto                --python_out=.
protoc object_detection\protos\preprocessor.proto                   --python_out=.
protoc object_detection\protos\region_similarity_calculator.proto   --python_out=.
protoc object_detection\protos\square_box_coder.proto               --python_out=.
protoc object_detection\protos\ssd.proto                            --python_out=.
protoc object_detection\protos\ssd_anchor_generator.proto           --python_out=.
protoc object_detection\protos\string_int_label_map.proto           --python_out=.
protoc object_detection\protos\train.proto                          --python_out=.
```

Install the python packages
```commandline
cd slim
python setup.py install
cd ..
python setup.py install
```
If you get the exception `error: could not create 'build': Cannot create a file when that file already exists` here, delete the `BUILD` file inside first

Now add the [source to the python path](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#add-libraries-to-pythonpath) or copy the `object_detection` folder and the `slim` folder into your `[Anaconda3]/Lib/site-packages` directory. 

# Dataset
If you are just interested in the dataset, the split and the annotations used in this project, you can run the following scripts to reproduce the dataset locally:

    cd keras_frcnn
    python muscima_image_cutter.py
    python DatasetSplitter.py
    
These two scripts will download the datasets automatically, generate cropped images along an Annotation.txt file and split the images into three reproducible parts for training, validation and test. 

# License

Published under MIT License,

Copyright (c) 2017 [Alexander Pacha](http://alexanderpacha.com), [TU Wien](https://www.ims.tuwien.ac.at/people/alexander-pacha)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
