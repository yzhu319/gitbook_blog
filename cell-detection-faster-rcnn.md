---
description: https://github.com/yzhu319/keras-frcnn
---

# Cell detection: faster-RCNN

## Objective

In this project, we want to train a model to automatically detect and classify different cells. This is a real-world use case of computer vision algorithms. The images are taken from a biomedical-engineering (BME) research lab.

The images captured under microscope contains CD8 (a T-cell receptor) and activated-CD8 (activated form of CD8, usually distorted in an irregular shape). These two cell types are shown in the following sample figure, which has 6 CD8 and 2 activated-CD8:

![Image\_sample](.gitbook/assets/image\_\_2021-06-28\_\_12-42-56.png)

The corresponding annotation information for this figure is saved in a text (or csv) file. Each row represent a cell with 6 columns:&#x20;

1. **image\_path:** contains the path of the image
2. **xmin:** x-coordinate of the bottom left part of the cell
3. **xmax:** x-coordinate of the top right part of the cell
4. **ymin:** y-coordinate of the bottom left part of the cell
5. **ymax:** y-coordinate of the top right part of the cell
6. **cell\_type:** denotes the type of the cell

The sample annotation info is shown as the table below:

| image\_path                 | x\_min | x\_max | y\_min | y\_max | cell\_type    |
| --------------------------- | ------ | ------ | ------ | ------ | ------------- |
| train\_images/Image\_sample | 113    | 1      | 150    | 27     | CD8           |
| train\_images/Image\_sample | 58     | 66     | 97     | 104    | CD8           |
| train\_images/Image\_sample | 324    | 111    | 370    | 158    | CD8           |
| train\_images/Image\_sample | 228    | 256    | 261    | 291    | CD8           |
| train\_images/Image\_sample | 312    | 359    | 352    | 397    | CD8           |
| train\_images/Image\_sample | 288    | 384    | 324    | 414    | CD8           |
| train\_images/Image\_sample | 124    | 258    | 156    | 303    | activated CD8 |
| train\_images/Image\_sample | 151    | 257    | 194    | 302    | activated CD8 |

With labeled training data, we aim to 1. detect the correct location of cells in a new image and 2. correctly classify two types of cells.

## Faster-RCNN for object detection

Object detection is a common task in computer vision. The algorithm we used here is Faster-RCNN. Faster-RCNN algorithm is optimized based on RCNN and Fast-RCNN, it replaces the selective search method with region proposal network (RPN) that makes the algorithm much faster.

## Implementation

The algorithm is realized with a Keras implementation of Faster R-CNN.

{% embed url="https://github.com/yzhu319/keras-frcnn" %}

The details can be found in the README of this repo.

## Results

With GPU (GeForce GTX 1660 Ti), we train a model using the training data sets (\~160 images). With each epoch, we record the following 4 metrics:

rpn\_cls, rpn\_regr, detector\_cls, detector\_regr

After about 50 epochs, the total error no longer decreases (the total-loss VS epochs curve plateaus) and we save the model parameters to a model file. Those parameters are then used to make inferences. The two sample predictions are as follows. We can see the model can correctly detect all the cells in the image, and correctly classify these two cell types!

![](<.gitbook/assets/cd8-sample1 (1).png>)

![](.gitbook/assets/cd8-sample2.png)



## Comments

* Cells are relatively small compared with image size, so we need to tune the box size of the anchor boxes. This is a critical parameter to tune in order to successfully "detect" the cells.
* Configuration with Tensorflow (1.0 version) and GPU takes some trial-and-error efforts (especially with Windows operating system), and the original faster-RCNN repo has minor bugs to be fixed. Configuration details are specified in "setting\_readme.txt".
* Making the code compile & run and generate prediction results is not a difficult task, the tricky part in implementing Deep Learning application to solve real-world problems is to find the right parameters that somehow 'magically' works. It takes multiple experiments.
* Algorithm is important, so is the input data. Image preprocessing and augmentation will greatly help with model performance-- the computer algorithm likes high-quality images obtained from well-trained experimentalists who know how to capture excellent cells sample under microscope.



