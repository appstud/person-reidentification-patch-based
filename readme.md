# Patch based person reidentification.
This is unofficial implementation in python of the CVPR 2013 paper Unsupervised salience learning for person re-identification.

## Installation
1) Install Python 3.6
2) Install Opencv, sklearn, imutils
3) Clone this repository
4) Download the dataset PRIDE and keep it in the root folder
4) The dataset can be downloaded from: 

   https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/prid11/

5) Extract descriptors by running:
   python $ROOT/extractDescriptors.py
6) Test the model by running:
   python $ROOT/matching.py --numberOfImageInGallery=20
    


**Citations:**

[1] Zhao R, Ouyang W, Wang X. Unsupervised salience learning for person re-identification. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2013 (pp. 3586–3593).
