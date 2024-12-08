# 1.    Acknowledgement
The main python files fork from the [repo](https://github.com/ferpb/spectralwaste-dataset.git). Thanks for ferpb's great work and sharing.

# 2. 	SpectralWaste Preprocessing:
(Not familiar, under learning)
For multimodal informations, when during the fusion process, it supposed to need alignment between RGB images and HSI images. The lib called [COTR](https://github.com/ubc-vision/COTR) could be helpful for this.
`spectralwaste/utils/image_processing.py` contains functions include resizing images, normalizing image data, and converting hyperspectral images into false color images.