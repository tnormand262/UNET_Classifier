**Summary**

The goal of this project is to take RGB geotiff files and use a neural network called [UNET]([url](https://arxiv.org/pdf/1505.04597)) to classify different objects within the image. It then outputs a shape file with polygons drawn around each object. The model was trained on publicly available images shared by the [ISPRS 2d-Semantic Labeling Contest]([url](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)) of Potsdam, Germany. This makes the model very good at identifying objects within an urban setting. The contest not only shared images, but also ground truth masks where all pixels that make up an object have already been assigned a class. Note: the original images and masks have a number of sync errors. The original images seem to be stitched together using some sort of blending between the patches which sometimes leads to curved or disjoint parts of the image. Open up the original images in QGIS to see for yourself.

**Dependencies**

GDAL, 
PyTorch, 
OpenCV, 
pillow, 
Albumentations, 
segmentation-models-pytorch

**How to Use**

1. Download the .ipynb files from the Github
2. Open the files in Jupyter Notebook, this can be done by installing Anaconda Navigator and opening Jupyter Notebooks from there. This will lead you to your file directory. Navigate to the relevant files.
3. If you don't want a new model skip to part 10
4. To train a new model you need to create a folder to store your dataset's patches. Open UnetTest.ipynb You can create these patches using create_patches(input_folder, output_folder). By default, the dataset is the Potsdam one. Ignore this if you are fine using the pre-generated patches.
5. Give each file path in parameters a corresponding endpoint
6. Modify hyperparameters (num_epochs, batch_size, etc.) I recommend 7 epochs though I've had good success with less. The default model is trained on 10 epochs.
7. Run all cells
8. Wait ~1 hour * num_epochs
9. Test the model at the bottom of the file. Simply give it a test image file path along with a ground truth mask. Change these in Testing Parameters. These need to be patches of the same size that the model was trained on. 
10. Transfer the model file_path to main.ipynb
11. Open main.ipynb
12. Ensure file paths in Parameters are configured and that the original geotiff file has uniform size images
13. Run all cells
14. Run main() and wait 45s-1m30s depending on size of image
15. Open the new image in QGIS.
16. Right click on the image in the Layers menu on the left hand side. Go to Properties
17. Change the viewing mode to Categorized
18. Select the drop down menu from Value
19. Select class
20. Select Classify from the bottom left
21. Close the Properties menu
22. Right click in the Layers menu and click on Zoom to Layer(s)

-----------------------------

**Model Information:**

The focus of this project was to start building out Whiteout Solutions' machine learning capabilities and so much of my time went into developing a model. I used the ground truth masks to teach the model what it was looking for and over the course of roughly 10 hours where it trained on 24,320 images, all of size 512x512 pixels. We obtain these masks by cropping the orignal image into smaller images of size 512 and then performing data augmentation and pre-processing steps using the Albumnetations python library that lets us increase the quality of our training set by performing random modifications of the images in the existing training set. It trained for 10 epochs, each containing 152 batches, and each batch contained 16 images. This took almost 10 hours and the model is only 80% accurate. Nevertheless, this proved to be sufficiently accurate for our purposes. We can visualize these errors by looking at the shape file in QGIS. The misclassified polygons are typically small and hardly noticeable. Further improvements to the model or the polygon creation function could alleviate these issues. Additionally, I used the segmentation-models-pytorch library to build this model. This library includes many different segmentation models and proved to be more accurate than my attempts to build Unet from scratch.

The construction of many ML models includes a downscaling and upscaling phase. This is where we perform the convolutions and is how we're able to accurately extract features that would have otherwise been difficult to see. The nature of this down/upscaling is to half/double the resolution of the image 4 times (see the Unet paper for why). The model requires the final resolution to be an integer which implies that we must have an original input divisible by 2^4 =32. This creates problems later. One way to solve this is to crop out the excess pixels. Ex. take an image of size 6000x6000. If we want to create patches of size 512x512 then we will only be able to create 11x11=121 patches for the image. This leaves behind pixels that aren't considered in the model at the right and bottom of the image (see the create_patches function for why). To solve this we use zero padding to add black pixels to the original geotiff and we effectively round up to the next dimension that is divisible by 512. Ex. In the previous example we couldn't divide 6000 by 512 without remainder. Using zero padding we add black pixels so that the geotiff becomes of size 6144x6144 which is perfectly divisible by 512 (6144/512=12). 

Unet and most other ML algorithms include a loss function and an optimizer. The loss function is one way we measure the quality of the the models predictions and the optimizer is how we tweak the models parameters between batches in order to minimize the loss function. The loss function I used was Multi-Dice Loss which is an extension of the Dice coefficient, which measures the overlap between predicted segmentation masks and ground truth masks. The Dice coeffecient, or Dice score, ranges from 0 to 1 with 1 being perfect and 0 being worst. Note that in some situations there are many ways to get the same score. Multi-Dice loss is the generalization of this to multiple classes. We effectively average the Dice scores between each class. Initially, I used segmentation models' crossentropyloss() function but faced severe missclassification issues. To resolve this, I adapted a Multi-Dice loss function that proved to be much better using [Followb1ind1y]([url](https://github.com/Followb1ind1y/Semantic-Segmentation-of-Aerial-Imagery))'s work as a jumping off point. Another loss function I tried was weighted cross entropy loss which considers the weights of the each class. We can find these fairly easily but because of the majority classes it's likely better to underweight the majority classes to avoid overfitting. When I tried this I obtained only mediocre results. The optimizer I used was Adam, a standard optimization algorithm used in many ML applications. 
Note: Multi-Dice loss has tended to give poor results in the first 2-3 epochs before obtaining higher results than the other loss functions I tried by the 7th epoch or so.

See main's "Initialize the model" section for how to create an instance of the model in detail.

The output of this model is a tensor that we convert to a numpy array and then to an image. Below, I've included an image showing the original patch, the ground truth mask, and the prediction. I will first include information on the classes. Note that the Unlabeled class seems to show up randomly. There is no rhyme or reason to it because the person who created these ground truth masks intended for it to be a catch all. Because of this, the model frequently labels construction, people, walls, and bare earth as Unlabeled. Adding to the number of classes could help this but it did not seem worthwhile to do so.

| Class        | Color |
| ------------- | ------------- |
| Vehicle       | Yellow  |
| Building  | Blue  |
| Vegetation       | Green  |
| Road       | White  |
| Ground  | cyan  |
| Unlabeled      | Red  |
![image](https://github.com/WhiteoutSolutions/planar_classification/assets/171065077/129c6175-5315-4804-8245-b1e59bdd40e4)
Note: The original image shown here belongs to the validation testset so the model has never "seen" this image before

The model does a great job preserving the features of the original image and arguably does a better job classifying the image than the ground truth mask labeled by a human!

**Polygon Extraction**

Now that we have a mask that has classified all the pixels as belonging to a class we're ready to use find_contours to draw edges on the borders of each object that was classified by the model. We then use these contours when we use a python library called shapely which has a tool for drawing polygons called Polygon(). This is how we're able to draw very accurate polygons around the objects.

**Main**

Lastly, I put it all together in a main function that takes a few parameters but mostly just file paths. For future users of the VM I doubt much needs to change but if this project is exported elsewhere the parameters section will need to be modified. I purposefully hard coded some of the parameters like patch_size because I think that the next step is to see how the model does on a new dataset. The main function includes all the necessary function calls to create a shape file with both the coordinate metadata and the polygon classifcations. First it reads the coordinate metadata from the geotiff and resizes the geotiff with zero padding. It then calls create_patches. Then it sends these patches to a temporary holding folder called patch_holder. patch_holder is emptied in between main function calls so no need to do anything there. The reason why it saves it to a folder and doesn't keep it as a list of numpy arrays is because when I tried to do that, it increased the main run time by 60+ seconds. It is likely that my implementation was inefficient but due to the imminent end of my time at Whiteout I did not explore further. 

These patches are then callled in predict_and_combine which returns a full mask that has the same dimensions as the original image as it removes the padding before returning the image. Then we call find_contours and then we use a for loop to give each polygon a class. I also included a class_colors attribute in the for loop and this is to show future users how to add new attributes should they choose to do so. The class_colors does nothing in QGIS, but you could choose to add new metadata. Additionally, there is a geo_to_pixel function that writes geolocation data to each pixel in the final image.

**Results**

Here are a few examples of geolocated shape files that the model has run on. The predictions are quite good and go to show that despite only 80% accuracy the model is still good enough for the job.
| Original        
| ------------- |
| ![image](https://github.com/WhiteoutSolutions/planar_classification/assets/171065077/46b5c035-e361-4c4f-91f6-eb86a3ebbc97)|
| Predicted Image |        
|![image](https://github.com/WhiteoutSolutions/planar_classification/assets/171065077/958e8c58-c775-4801-a1a3-3df8e20cc3ca)|
| Original        |
| ![image](https://github.com/WhiteoutSolutions/planar_classification/assets/171065077/4afaa051-c1cd-41f8-b51d-886fa42be054)|
| Predicted Image |        
|![image](https://github.com/WhiteoutSolutions/planar_classification/assets/171065077/41107b52-4034-4a36-ad80-b20ed8f54066)|
Note: The model has never been shown these images before and is classifying each image in under 1min30s on the VM


**Future Improvements**

In my limited testing the model does poorly when the camera zoom and camera altitude differ from those of the Potsdam dataset. When I imported images that did not respect the zoom or altitude specifications like the [Dubai dataset]([url](https://www.kaggle.com/datasets/humansintheloop/semantic-segmentation-of-aerial-imagery)) I frequently found that the model performed poorly. To address this issue, an even richer dataset is required. One that includes variations in height, zoom, time of day, time of year, etc. Doing this would likely require a blend of drone and satellite images. In speaking to Yanick and Jack it seems feasible to write a program that takes pictures of a random place in the world at random zoom levels using satellite imagery but it's not something I've had time to explore. 

The model was trained and tested on images with square dimensions. Much of the code was written with this assumption in mind and in the future, the model should be generalized to address this.

An easier fix is how the program stores intermediate files. I think that saving the individual patches and then predicting them and then saving them again is likely inefficient and is an artifact of when I built out the code and needed to test each function one at a time. 

**Important Information**

The model itself is stored here: \\192.168.1.12\ResearchDevelopment\AI_ObjectClassification\Models\ObjClassification1\Saved Models\ScratchUnet5.pth

If you try and copy entire folders into a new directory Windows will sometimes create a file within the folder called thumbs.db which interfers with the training of the model. You can remove it to solve the issue.

**What I Learned**

This project has initiated me in the world of machine learning and given me a new interest that I hope to pursue into the future. I was challenged to build a complex model based on a mathematical operation I was unfamiliar with and was able to realize the goals of this project. Specifically, I learned about Unet, loss functions, and was exposed to a myriad of python libraries I was unfamilair with.

**Helpful Links**

Followb1ind1y's work: https://github.com/Followb1ind1y/Semantic-Segmentation-of-Aerial-Imagery/blob/main/Semantic_Segmentation_of_Aerial_Imagery.ipynb

Unet Paper: https://arxiv.org/pdf/1505.04597

Potsdam Data Download (Select the Potsdam folder): https://www.kaggle.com/datasets/aletbm/urban-segmentation-isprs/data?select=Potsdam 



