﻿$pathToSourceRoot = "C:\Users\Alex\Repositories\MusicObjectDetector\"
$pathToTranscript = "$($pathToSourceRoot)"

# Allowing wider outputs https://stackoverflow.com/questions/7158142/prevent-powergui-from-truncating-the-output
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 9999
$newsize.width = 1500
$pswindow.buffersize = $newsize

cd $pathToSourceRoot
echo "Appending source root $($pathToSourceRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $pathToSourceRoot

#python C:\Users\Alex\Repositories\MusicSymbolClassifier\ModelTrainer\tests\Symbol_test.py
#cd "tests"
#pytest


#######################################
# Build tools for data_generators_fast
#######################################
cd keras_frcnn
cd py_faster_rcnn
python setup.py build_ext --inplace
cd ..
cd ..

################################################
# Upcoming Trainings 
################################################

# Started on Donki, 06.09.2017
$configuration_name = "streched_anchor_box_ratios"
$base_name = "2017-09-06"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs 5
Stop-Transcript

$configuration_name = "small_anchor_box_scales"
$base_name = "2017-09-06"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs 5
Stop-Transcript

$configuration_name = "many_anchor_box_scales"
$base_name = "2017-09-06"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs 5
Stop-Transcript


#######################################################
# Below are configurations that already were 
# started on a machine and should not run again, 
# so we will terminate this PS-script here
# but retain those configurations for documentation
#######################################################
exit

# Started on Donki, 05.09.2017
Start-Transcript -path "$($pathToTranscript)2017-09-05_resnet50.txt" -append
python "$($pathToTranscript)TrainModel.py" --recreate_dataset_directory --network resnet50 --output_weight_path "2017-09-05_resnet50.hdf5" --config_filename "2017-09-05_config.pickle"
Stop-Transcript
