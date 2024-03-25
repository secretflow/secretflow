# Image segmentation task based the vision BASNet

## Introduction

This example demonstrates how to use SecretFlow to train BASNet to segment the image based TensorFlow;

## Run
### get dataset
When you run the example at the first time, you are supposed to run the shell script to download the dataset.
```
sh get_dataset.sh
```
### run the script
run the script to train BASNet model.
```
python tensorflow_basnet_image_segmentation.py
```

## References
- [Highly accurate boundaries segmentation using BASNet](https://keras.io/examples/vision/basnet_segmentation/)