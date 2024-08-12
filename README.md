**Summary**

The goal of this project is to take RGB geotiff files and use a neural network called [UNET]([url](https://arxiv.org/pdf/1505.04597)) to classify different objects within the image. It then outputs a shape file with polygons drawn around each object. The shape file also preserves the perspective and geolocation metadata so that the shape file can be perfectly overlayed with a street map. In the results I've included I kept part of the street map in the background to demonstrate this. The model was trained on publicly available images shared by the [ISPRS 2d-Semantic Labeling Contest]([url](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)) of Potsdam, Germany. This makes the model very good at identifying objects within an urban setting. The contest not only shared images, but also ground truth masks where all pixels that make up an object have already been assigned a class color. What differentiates my work from the majority of similar projects is that others successfully segmented an image and produced an RGB output with relabeled pixels. My work goes one step further and adds new data to an image and allows for objects of speciifc classes to be viewed in isolation. This is advantageous for applications like searching for minority classes. Lastly, I developed this program whilst interning at a company and as such will not be releasing any of the associated code. The company kindly agreed to let me share my results.

Note: the original images and masks have a number of sync errors. The original images seem to be stitched together using some sort of blending between the patches which sometimes leads to curved or disjoint parts of the image and I was not able to correct these mistakes.

**Dependencies**

GDAL, 
PyTorch, 
OpenCV, 
pillow, 
Albumentations, 
segmentation-models-pytorch
-----------------------------

**Model Information:**

The focus of this project was to start building out Whiteout Solutions' machine learning capabilities and so much of my time went into developing a model. I used the ground truth masks to teach the model what it was looking for and over the course of roughly 10 hours it trained on 24,320 images, all of size 512x512 pixels. We obtain these masks by cropping the orignal images from the Potsdam dataset (6000 by 6000) into smaller images of size 512 and then performing data augmentation and pre-processing steps using the Albumnetations python library that lets us increase the quality of our training set by performing random modifications of the images in the existing training set. It trained for 10 epochs, each containing 152 batches, and each batch contained 16 images. This took almost 10 hours and the model is only 80% accurate. Nevertheless, this proved to be sufficiently accurate for our purposes. We can visualize these errors by looking at the shape file in QGIS. The misclassified polygons are typically small and hardly noticeable. Additionally, I used the segmentation-models-pytorch library to build this model. This library includes many different segmentation models and proved to be more accurate than my attempts to build UNET from scratch.

The construction of many ML models includes a downscaling and upscaling phase. This is where we perform the convolutions and is how we're able to accurately extract features that would have otherwise been difficult to see. The nature of this down/upscaling is to half/double the resolution of the image 4 times (see the UNET paper for why). The model requires the final resolution to be an integer which implies that we must have an original input divisible by 2^4 =32. This creates problems later when we want to run the model on images that don't fit those dimensions. One way to solve this is to crop out the excess pixels. Ex. take an image of size 6000x6000. If we want to create patches of size 512x512 then we will only be able to create 11x11=121 patches for the image. This leaves behind pixels that aren't considered in the model at the right and bottom of the image since the patch creation moves left to right, top to bottom. To solve this we use zero padding to add black pixels to the original geotiff and we effectively round up to the next dimension that is divisible by 512. Ex. In the previous example we couldn't divide 6000 by 512 without remainder. Using zero padding we add black pixels so that the geotiff becomes of size 6144x6144 which is perfectly divisible by 512 (6144/512=12). 

UNET and most other ML algorithms include a loss function and an optimizer. The loss function is one way we measure the quality of the the models predictions and the optimizer is how we tweak the models parameters between batches in order to minimize the loss function. The loss function I used was Multi-Dice Loss which is an extension of the Dice coefficient, which measures the overlap between predicted segmentation masks and ground truth masks. The Dice coeffecient, or Dice score, ranges from 0 to 1 with 1 being perfect and 0 being worst. Note that in some situations there are many ways to get the same score. Multi-Dice loss is the generalization of this to multiple classes. We effectively average the Dice scores between each class. Initially, I used segmentation models' crossentropyloss() function but faced severe missclassification issues. To resolve this, I adapted a Multi-Dice loss function that proved to be much better using [Followb1ind1y]([url](https://github.com/Followb1ind1y/Semantic-Segmentation-of-Aerial-Imagery))'s work as a jumping off point. Another loss function I tried was weighted cross entropy loss which considers the weights of the each class. We can find these fairly easily but because of the majority classes it proved better to underweight the majority classes to avoid overfitting. When I tried this I obtained only mediocre results. The optimizer I used was Adam, a standard optimization algorithm used in many ML applications. 
Note: Multi-Dice loss has tended to give poor results in the first 2-3 epochs before obtaining higher results than the other loss functions I tried by the 7th epoch or so.

Below, I've included an image showing the original patch, the corresponding ground truth mask, and the prediction. I will first include information on the classes. Note that the Unlabeled class seems to show up randomly. There is no rhyme or reason to it because the person who created these ground truth masks intended for it to be a catch all. Because of this, the model frequently labels construction, people, walls, and bare earth as Unlabeled. Adding to the number of classes could help this but for my application this was not necessary.

| Class        | Color |
| ------------- | ------------- |
| Vehicle       | Yellow  |
| Building  | Blue  |
| Vegetation       | Green  |
| Road       | White  |
| Ground  | Cyan  |
| Unlabeled      | Red  |
![image](https://github.com/tnormand262/UNET_Classifier/assets/160414926/eb423e00-4cf2-441d-a0af-f7fedf418f80)
Note: The original image shown here belongs to the validation testset so the model has never "seen" this image before. The model runs on patches of this size in less than 5 seconds.

The model does a great job preserving the features of the original image and arguably does a better job classifying the image than the ground truth mask labeled by a human!

**Polygon Extraction**

Now that we have a mask that has classified all the pixels as belonging to a class we're ready to draw the edges between each object that was classified by the model. I used these contours in conjunction with a python library called shapely which has a tool for drawing polygons called Polygon(). This is how we're able to draw very accurate polygons around the objects.

**Main**

Lastly, I created a cell in my jupyter notebook where I can hardcode file paths without having to change them manually later on. The main function automates the entire process of reading the original tiff file and its metadata, creating the patches to run the model on, running the model, and then stitching the predicted masks back together. It then extracts polygons and adds the data to a shape file which is put into an output directory so that we can use QGIS to visualize the output.

**Results**

Here are a few examples of geolocated shape files that the model has run on. The predictions are quite good and go to show that despite only 80% accuracy, the model is still very good at its job. I estimate that the final output is 95% accurate for medium sized objects and 90% accurate for larger objects like buildings or grass.
| Original        
| ------------- |
| ![image](https://github.com/tnormand262/UNET_Classifier/assets/160414926/387db528-1c59-45de-8256-efb8012bee8b)
| Predicted Image |        
|![image](https://github.com/tnormand262/UNET_Classifier/assets/160414926/dbbaa830-2cca-4839-bf79-86f17766de3b)
| Original        |
|![image](https://github.com/tnormand262/UNET_Classifier/assets/160414926/98969a2f-51c8-4c58-aebc-9a26522ce398)|
| Predicted Image |        
|![image](https://github.com/tnormand262/UNET_Classifier/assets/160414926/df208fec-b1e2-41ff-ae8c-c4c910cde303)|
Note: The model has never been shown these images before and is classifying each image in under 1min30s on the VM


**Future Improvements**

In my testing the model does poorly when the camera zoom and camera altitude differ from those of the Potsdam dataset. When I imported images that did not respect the zoom or altitude specifications like the [Dubai aerial imagery dataset]([url](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)) I frequently found that the model performed poorly. To address this issue, an even richer dataset is required. One that includes variations in height, zoom, time of day, time of year, etc. Doing this would likely require a blend of drone and satellite images. It seems feasible to write a program that stores images of a random places in the world at random zoom levels using satellite imagery but it's not something I've explored yet. 

The model was trained and tested on images with square dimensions. Much of the code was written with this assumption in mind and in the future, the model should be generalized to address this.

**What I Learned**

This project has introduced me to the world of machine learning and given me a new interest that I hope to pursue into the future. I was challenged to build a complex model based on a mathematical operation I was unfamiliar with and was able to realize the goals of this project. Specifically, I learned about UNET, loss functions, and was exposed to a myriad of python libraries I was unfamilair with.

**Helpful Links**

Followb1ind1y's work: https://github.com/Followb1ind1y/Semantic-Segmentation-of-Aerial-Imagery/blob/main/Semantic_Segmentation_of_Aerial_Imagery.ipynb

UNET Paper: https://arxiv.org/pdf/1505.04597

Potsdam Data Download (Select the Potsdam folder): https://www.kaggle.com/datasets/aletbm/urban-segmentation-isprs/data?select=Potsdam 



