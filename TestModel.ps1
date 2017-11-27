$pathToSourceRoot = "C:\Users\Alex\Repositories\MusicObjectDetector\"
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

Start-Transcript -path "$($pathToTranscript)TestResults.txt" -append

$Vgg16_configurations = "2017-11-24_Vgg16_muscima_pp_1","2017-11-25_Vgg16_muscima_pp_2","2017-11-26_Vgg16_muscima_pp_3","2017-11-27_600-region-proposal_Vgg16_muscima_pp_1","2017-11-08_800-rpns_0.7-overlap_vgg_small_anchor_box_scales","2017-11-07_600-rpns_0.7-overlap_vgg_small_anchor_box_scales_many_rois","2017-11-03_600-rpns_0.7-overlap_vgg_small_anchor_box_scales"

$ResNet50_configurations = "2017-11-13_1200-rpns_0.7-overlap_resnet50_small_anchor_box_scales","2017-11-11_1200-rpns_0.7-overlap_resnet50_small_anchor_box_scales","2017-11-10_1000-rpns_0.7-overlap_resnet50_small_anchor_box_scales","2017-11-09_800-rpns_0.7-overlap_resnet50_small_anchor_box_scales","2017-11-03_600-rpns_0.7-overlap_resnet50_small_anchor_box_scales"

$SimpleResNet_configurations = "2017-11-12_800-rpns_0.7-overlap_simple_resnet_small_anchor_box_scales"

$model_name = "Vgg16"
foreach ($configuration in $Vgg16_configurations) {
    python "$($pathToTranscript)TestModel.py" --testdata_path data/test --num_rois 32 --config_path "results/$($configuration).pickle" --model_path "results/$($configuration).hdf5" --model_name $model_name --non_max_suppression_max_boxes 300 --non_max_suppression_overlap_threshold 0.7 --classification_accuracy_threshold 0.4
}

$model_name = "ResNet50"
foreach ($configuration in $ResNet50_configurations) {
    python "$($pathToTranscript)TestModel.py" --testdata_path data/test --num_rois 32 --config_path "results/$($configuration).pickle" --model_path "results/$($configuration).hdf5" --model_name $model_name --non_max_suppression_max_boxes 300 --non_max_suppression_overlap_threshold 0.7 --classification_accuracy_threshold 0.4
}

$model_name = "SimpleResNet"
foreach ($configuration in $SimpleResNet_configurations) {
    python "$($pathToTranscript)TestModel.py" --testdata_path data/test --num_rois 32 --config_path "results/$($configuration).pickle" --model_path "results/$($configuration).hdf5" --model_name $model_name --non_max_suppression_max_boxes 300 --non_max_suppression_overlap_threshold 0.7 --classification_accuracy_threshold 0.4
}

# Try different values for one single configuration
$model_name = "Vgg16"
$configuration = "2017-11-26_Vgg16_muscima_pp_3"
foreach ($number_of_rois in 32,64,256) 
{
    foreach ($non_max_suppresion_max_boxes in 300,600,1200) 
    {
        foreach ($non_max_suppression_overlap_threshold in 0.6, 0.7, 0.8) 
        {
            #Write-Output "$($pathToTranscript)TestModel.py" --testdata_path data/test --num_rois $number_of_rois --config_path "results/$($configuration).pickle" --model_path "results/$($configuration).hdf5" --model_name $model_name --non_max_suppression_max_boxes $non_max_suppresion_max_boxes --non_max_suppression_overlap_threshold $non_max_suppression_overlap_threshold --classification_accuracy_threshold 0.4
            python "$($pathToTranscript)TestModel.py" --testdata_path data/test --num_rois $number_of_rois --config_path "results/$($configuration).pickle" --model_path "results/$($configuration).hdf5" --model_name $model_name --non_max_suppression_max_boxes $non_max_suppresion_max_boxes --non_max_suppression_overlap_threshold $non_max_suppression_overlap_threshold --classification_accuracy_threshold 0.4
        }
    }
}

Stop-Transcript
