# 1.	SpectralWaste Format Analysis:
The SpectralWaste dataset contains sequences of multimodal RGB and hyperspectral images.Each image is identified with the date, number of sequence and time it was captured, with the format yyyymmdd-xx-HHMMSS.
```
|dataset
| +-- 20220928
|      +-- 01
|           +-- rgb
|                +-- 20220928_01_105147.jpg
|           +-- hyper
|                +-- 20220928_01_105147.tiff
```
The dataset contains 7655 multimodal images, and 852 of those are annotated with object segmentation masks. The Raw labeled RGB and HIS dataset is [105GB]( https://unizares-my.sharepoint.com/personal/756012_unizar_es/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F756012%5Funizar%5Fes%2FDocuments%2Fspectralwaste%2Flabeled%5Fdataset&ga=1), and a simplified version with [27GB]() provided. The annotations are stored following the COCO format for object detection. The JSON files, one containing annotations for the RGB images and another with a reduced set of annotations for the hyperspectral images. 

# 2. 	SpectralWaste Preprocessing:
(Not familiar, under learning)
For multimodal informations, when during the fusion process, it supposed to need alignment between RGB images and HSI images. The lib called [COTR](https://github.com/ubc-vision/COTR) could be helpful for this.
`dataset/spectralwaste/utils/image_processing.py` contains functions include resizing images, normalizing image data, and converting hyperspectral images into false color images.

# 3.    HSI and RGB Multimodal Detection Task:
For the object detection method using the fusion of hyperspectral and RGB images, the main idea is to integrate features from both types of images. The key challenge lies in aligning the features of the two images and leveraging the combined features to accomplish the detection task.

As this is my first try for multimodal fusion, ideally I want to follow the method provided by [CFT](https://arxiv.org/pdf/2111.00273), parellarly process the RGB and HSI data in two Conv based branch and during each stage of the hierachical output, concat the output of two modals' Conv branch as a feature map and apply MHSA method to extract the fusion features. Aforementioned are about the backbone design part, I'm still learning about the neck design part but maybe keep the same as previous Detection method is enough.

If so, the key technical challenge in multimodal fusion of HSI and RGB images lies in the alignment of image features, while the remaining issues, based on my current understanding, should be similar to those encountered in ordinary object detection tasks.

# 4.    Potential improvements
I have limited knowledge in the alignment of multimodal information and am unable to provide optimization solutions in this area. Therefore, based on my past experience in object detection tasks, I will offer some optimization insights regarding feature learning and fusion:

## **Optimizing the Backbone:**
- Given that the targets in the spectralwaste dataset are usually large, there is no need for excessive operations on high-level feature maps. Instead, more emphasis can be placed on feature extraction and fusion in stage 3 (P4) and stage 4 (P5), or even consider adding a stage 5 (P6) to further deepen the feature maps.
- The backbone module in the CMF paper can be improved. The Attention Block for feature fusion can be further enhanced, for example, by adding additional processing for the extraction of K and V.
## **Optimizing the Neck:**
The CMF paper maintains the design of YOLOv5, but it can be considered to incorporate the structure of DETR to enhance global vision. Further optimizations can be made based on this structure.
## **Optimizing the Conv Module:**
There is room for optimization in the C3 and RepC3 modules of the YOLO series.