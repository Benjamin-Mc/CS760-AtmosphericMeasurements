# CS760-AtmosphericMeasurements

This project aims to use camera images for the prediction of atmospheric measurements such as temperature and relative humidity. 

This project was complete for the course COMPSCI760 at The University of Auckland, by group P13. 

This group was comprised of the members: Benjamin Mclntosh, Chenchen Ning, Chunnan Li, Kevin Cham, and Xiaowei Li.

The "CNN_Model_Template.ipynb" file contains a simple template for training a CNN on our dataset. This takes care of the immediate image preprocessing and transformations, model training and test predictions, as well as some simple results analysis.
Users can simply select which of the two measurements they hope to predict, the image resolution, and whether the images should be in colour or greyscale. Note that the CNN model parameters do also need to be adjusted to the specified image size, and number of output predictions.
In addition, the trained model is saved in a .sav file, and the predictions are saved in a .csv file. Be sure to rename these files as desired.

The "Baseline Models.ipynb" is also used to construct our two baseline models.
In the Results folder, there is a simple "Results Analysis.ipynb" file, and the saved predictions and HTML files from the training of each specific model.

The dataset we use is a collection of roadside images from Poland. This is a link to the entire dataset. https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/SV9N9F.
We use a final dataset of 7718 images. Given the size of the image dataset we use, we cannot store it on Github, and UoA plans to restrict the size of the Google Drive space, so they are stored in a personal Drive that can be found here.
https://drive.google.com/drive/folders/1139r6vKUHqqNfAw_BZXg2GXk2STQHlNP?usp=sharing

To easily view the HTML files, either download or paste the URL of the file on this website.
https://htmlpreview.github.io/
