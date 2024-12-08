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