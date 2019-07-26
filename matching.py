import scipy.io as sio
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from skimage import feature
import numpy as np
import os
import imutils
from sklearn.neighbors import NearestNeighbors
import scipy
from os import listdir
import random
 
def colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r,g,b)) 
  return ret

def matchPatch(patchDescriptor,rowOfThePatch, gallery, numberOfPatchInARowOfImage,m,K, sigma=100.0):
    ###patchDescriptor: the descriptor of a patch
    ###gallery: matrix containing all the patch descriptors of all images in the gallery
    ###m: the adjacency constraint parameter
    ###numberOfPatchInARowOfImage: int representing number Of Patch In an Image Row

    _,numberOfPatchInImage,numberOfImagesInGallery=gallery.shape
    minIndexForSearch=max(0,(rowOfThePatch-m)*numberOfPatchInARowOfImage)
    maxIndexForSeach=min((rowOfThePatch+m)*numberOfPatchInARowOfImage,numberOfPatchInImage)
    
    distanceListToAllImages=[]
    similarityMeasureInEachImage=np.zeros([1,numberOfImagesInGallery])
    arrayOfIndices=np.zeros([1,numberOfImagesInGallery])
    
    for imageIndex in range(numberOfImagesInGallery):
       
        nbrs = NearestNeighbors(n_neighbors=1,metric='l2').fit(gallery[:,minIndexForSearch:maxIndexForSeach+numberOfPatchInARowOfImage,imageIndex].T)       
        distance, indice = nbrs.kneighbors(patchDescriptor.reshape(1,-1))
        
        similarityMeasure=np.exp(-distance**2/(2*sigma**2))
        distanceListToAllImages.append(distance)
        similarityMeasureInEachImage[0,imageIndex]=similarityMeasure
        arrayOfIndices[0,imageIndex]=int(indice+minIndexForSearch)
        
    distanceListToAllImages.sort()
    saliencyScore=distanceListToAllImages[int(K)]
    
    return saliencyScore, similarityMeasureInEachImage, np.asarray(arrayOfIndices,np.uint16)

cv2.namedWindow("image stitched",cv2.WINDOW_NORMAL)

def computeSimilarityBetweenImagesAndDraw(descriptorsOfPatchesInImageToMatch, gallery, numberOfPatchInARow, saliencyScoreOfGallery,imageGallery,imageTest,path,indexOfImage,alpha=1.0,sigma=0.3):
    ###saliencyScoreOfGallery: each row is an image, column is a patch in an image
    
    NImages=5
    #NPatches=4
    #colorsPatches=colors(NPatches)
    colorsPatches=[(0,255,0),(255,0,0),(0,0,255),(0,0,0)]
    _,numberOfPatches,numberOfImageInGallery=gallery.shape
    arrayOfAllPatchIndices=np.zeros([numberOfPatches,numberOfImageInGallery])
    similarityMatrix=np.zeros([numberOfPatches,numberOfImageInGallery])
    ####
    saliencyScoreOfImage=np.zeros(numberOfPatches)
    ###
    for patchNum in range(numberOfPatches):
        saliencyScore, similarityMeasure, arrayOfIndices=matchPatch(descriptorsOfPatchesInImageToMatch[:, patchNum], int(patchNum/numberOfPatchInARow), gallery, numberOfPatchInARow , m=2, K=numberOfImageInGallery/2,sigma=sigma)
        ###
        saliencyScoreOfImage[patchNum]=saliencyScore
        arrayOfAllPatchIndices[patchNum,:]=arrayOfIndices
        ###
        saliencyScoreOfCorrespondantPatches=saliencyScoreOfGallery[range(numberOfImageInGallery),arrayOfIndices[0,:]]
         
        similarityMatrix[patchNum,:]=(saliencyScore*similarityMeasure*saliencyScoreOfCorrespondantPatches)/((alpha+np.abs(saliencyScore-saliencyScoreOfCorrespondantPatches)))
        
    similarityScores=np.sum(similarityMatrix,0)  
    indexOfImagesToReturn=similarityScores.argsort()[::-1]      
    indexOfImages=indexOfImagesToReturn[0:NImages]
    indexOfTheMostSalientPatches=[]
    
    for i in range(2,int(numberOfPatches/numberOfPatchInARow)-2,8):
        indexOfTheMostSalientPatches.append(i*numberOfPatchInARow+similarityMatrix[i*numberOfPatchInARow:(i+1)*numberOfPatchInARow,indexOfImagesToReturn[0]].argsort()[::-1][0])
       
    indexOfTheMostSalientPatches=np.array(indexOfTheMostSalientPatches)
    patchSize=10
    numberOfPatchInAColumn=30
    height,width,_=imageTest.shape
    x_grid=np.linspace(patchSize/2,width-patchSize/2+1,numberOfPatchInARow)
    y_grid=np.linspace(patchSize/2,height-patchSize/2+1,numberOfPatchInAColumn)
    imageStitched=np.asarray(imageTest,np.uint8)
    verticalBlank=255*np.ones([imageTest.shape[0],10,3],dtype=np.uint8)
    
    for ind,patchInd in enumerate(indexOfTheMostSalientPatches):
        xTest,yTest=x_grid[int(patchInd%numberOfPatchInARow)],y_grid[int(patchInd/numberOfPatchInARow)]
        cv2.rectangle(imageStitched,(int(xTest-patchSize/2),int(yTest-patchSize/2)),(int(xTest+patchSize/2),int(yTest+patchSize/2)),colorsPatches[ind] ,2)
        
    for imgIndex in indexOfImages:
        img=np.asarray(imageGallery[imgIndex,:,:,:],np.uint8)

        for ind,patchInd in enumerate(indexOfTheMostSalientPatches):
            
            patchNumInImage=arrayOfAllPatchIndices[patchInd,imgIndex]
            x,y=x_grid[int(patchNumInImage%numberOfPatchInARow)],y_grid[int(patchNumInImage/numberOfPatchInARow)]
            cv2.rectangle(img,(int(x-patchSize/2),int(y-patchSize/2)),(int(x+patchSize/2),int(y+patchSize/2)), colorsPatches[ind],2)

        imageStitched=np.hstack((np.hstack((imageStitched,verticalBlank)),img))
            
    cv2.imshow("image stitched",imageStitched)
    k=cv2.waitKey(1)
    cv2.imwrite(os.path.join(path,"matching"+str(indexOfImage)+'.jpg'), imageStitched)  
            
    return indexOfImagesToReturn+1
    
def computeSimilarityBetweenImages(descriptorsOfPatchesInImageToMatch, gallery, numberOfPatchInARow, saliencyScoreOfGallery,alpha=5.0,sigma=100):
    ###saliencyScoreOfGallery: each row is an image, column is a patch in an image
    
    _,numberOfPatches,numberOfImageInGallery=gallery.shape
    similarityMatrix=np.zeros([numberOfPatches,numberOfImageInGallery])
    
    for patchNum in range(numberOfPatches):
        saliencyScore, similarityMeasure, arrayOfIndices=matchPatch(descriptorsOfPatchesInImageToMatch[:, patchNum], int(patchNum/numberOfPatchInARow), gallery, numberOfPatchInARow , m=2, K=numberOfImageInGallery/2,sigma=sigma)
        saliencyScoreOfCorrespondantPatches=saliencyScoreOfGallery[range(numberOfImageInGallery),arrayOfIndices[0,:]]
        similarityMatrix[patchNum,:]=(saliencyScore*similarityMeasure*saliencyScoreOfCorrespondantPatches)/((alpha+np.abs(saliencyScore-saliencyScoreOfCorrespondantPatches)))
    similarityScores=np.sum(similarityMatrix,0)
    
    return similarityScores.argsort()[::-1]

def computeSaliencyScoresOfImageGallery(descriptorsOfPatchesInImageToMatch, gallery, numberOfPatchInARow):
    ###saliencyScoreOfGallery: each row is an image, column is a patch in an image
    
    _,numberOfPatches,numberOfImageInGallery=gallery.shape
   
    saliencyScoreArray=np.zeros([1,numberOfPatches])
    for patchNum in range(numberOfPatches):
        saliencyScore, similarityMeasure, arrayOfIndices=matchPatch(descriptorsOfPatchesInImageToMatch[:, patchNum], int(patchNum/numberOfPatchInARow), gallery, numberOfPatchInARow , m=2, K=numberOfImageInGallery/2)
        saliencyScoreArray[0,patchNum]=saliencyScore

    return saliencyScoreArray

def stackSaliencyImages(saliencyGallery,newSaliency,img):
    
    saliencyOfImg=255.0*((newSaliency-np.min(newSaliency))/np.float32(np.max(newSaliency)))
    saliencyOfImg=np.asarray(scipy.misc.imresize(newSaliency, (128,64), interp='bilinear'),dtype=np.uint8)
    
    coloredSaliencyOfImg=cv2.applyColorMap(saliencyOfImg, cv2.COLORMAP_JET)
   
    
    #img=cv2.imread(pathImages[imageNumber])
    blankHorizontal=255*np.ones([10,img.shape[1],3],dtype=np.uint8)
    saliencyOriginalImage=np.vstack((np.vstack((img,blankHorizontal)),coloredSaliencyOfImg))
    cv2.imshow('saliency',  saliencyOriginalImage)
    #cv2.imshow('saliency', saliencyOriginalImage)
    #cv2.imwrite(os.path.join(  pathToSaveImage,str(imgIndex+1)+'.jpg'), saliencyOriginalImage)
    if(saliencyGallery is None):
        return saliencyOriginalImage
    else:
        
        blankVertical=255*np.ones([img.shape[0]*2+blankHorizontal.shape[0],blankHorizontal.shape[0],3],dtype=np.uint8)
        return np.hstack((np.hstack((saliencyGallery,blankVertical)),saliencyOriginalImage))
    
    
def readImages(path):

    imageGallery=np.zeros([len(path),128,64,3])
    
    for imgCounter,imgPath in enumerate(path):       
        imageGallery[imgCounter,:,:,:]=cv2.imread(imgPath)
        
    return  imageGallery

def readDescriptors(numCam,args):
    pathOfCam=os.path.join(args.data_root,'cam'+str(numCam))
    pathOfData=os.path.join( pathOfCam,str(numCam)+'.mat')
    matFile=sio.loadmat(pathOfData)
    gallery=matFile["Gallery"]
    personIDGallery=matFile["personID"]

    path_maker = lambda dir_name: os.path.join(pathOfCam, dir_name)
    directoryNames=os.listdir(pathOfCam)
    directoryNames=list(map(path_maker,directoryNames))
    images=readImages(directoryNames)
  
    return gallery,personIDGallery,images
  
def main(args):
   
    gallery,personIDGallery,galleryImages=readDescriptors("_a",args)
    galleryTest,personIDGalleryTest,testImages=readDescriptors("_b",args)
    try:
        gallery=gallery[:,:,0:args.numberOfImageInGallery]
        galleryTest=galleryTest[:,:,0:args.numberOfImageInGallery]
    except:
        print("There is not ",args.numberOfImageInGallery," in gallery, using the available images")
        
    saliencyOfGallery=None
    saliencyGallery=None
    numberOfImagesToSaveForSaliencyMap=2

    gridStep=args.grid_step
    patchSize=args.patchSize

    widthOfImages=galleryImages.shape[2]

    numberOfPatchInARow=len(np.arange(patchSize/2,widthOfImages-patchSize/2+1,gridStep))
    
    try:
        pathToSaveImage=os.path.join(args.data_root,"SaliencyOfGallery")    
        os.makedirs(pathToSaveImage)    
    except:
        print("Folder SaliencyOfGallery already exist!")
        
    try:
        pathToSaveImagePrediction=os.path.join(args.data_root,"Prediction")
        os.makedirs(pathToSaveImagePrediction)
    except:
        print("Folder prediction already exist!")
      
    
    
    for imgIndex in range(gallery.shape[2]): 
        saliencyOfImg=computeSaliencyScoresOfImageGallery(gallery[:,:,imgIndex], gallery,numberOfPatchInARow)
        if(saliencyOfGallery is None):
            saliencyOfGallery=saliencyOfImg
        else:
            saliencyOfGallery=np.vstack((saliencyOfGallery,saliencyOfImg))
          
        saliencyGallery=stackSaliencyImages(saliencyGallery,saliencyOfImg.reshape(-1,numberOfPatchInARow),np.asarray(galleryImages[imgIndex,:,:,:],np.uint8))
        if(imgIndex%numberOfImagesToSaveForSaliencyMap==0 and imgIndex!=0):
            cv2.imwrite(os.path.join(pathToSaveImage,str(imgIndex+1)+'.jpg'), saliencyGallery)
            saliencyGallery=None       
        
        
    
    similarityMatrix=np.zeros([galleryTest.shape[2],gallery.shape[2]])
    sigmas=[0.1,0.2]
    alphas=[0.8,0.9,1]
    maxAccRank=0
    maxSigma=0
    maxAlpha=0
    bestAccArray=[]
    
    for sigma in sigmas:
        for alpha in alphas:
            print("\n Testing with alpha=",alpha," and sigma=",sigma)
            
            for imgIndex in range(galleryTest.shape[2]):
                similarityMatrix[imgIndex,:]=computeSimilarityBetweenImagesAndDraw(galleryTest[:,:,imgIndex], gallery,numberOfPatchInARow, saliencyOfGallery,galleryImages,testImages[imgIndex,:,:,:],indexOfImage=imgIndex,alpha=alpha,sigma=sigma,path=pathToSaveImagePrediction)
                ##similarityMatrix[imgIndex,:]=computeSimilarityBetweenImages(galleryTest[:,:,imgIndex], gallery,14, saliencyOfGallery,sigma=sigma,alpha=alpha)+1
                print( "Similarity array for image",imgIndex+1,"is:"+str(similarityMatrix[imgIndex,:]))
            
            ranks=[1,3,5,10]
            accRank=0
            accArray=[]
            for rank in ranks:
                acc=0.0
                for imgIndex in range(galleryTest.shape[2]):
                    if(imgIndex in list(similarityMatrix[imgIndex,0:rank]-1)):
                        acc=acc+1

                accArray.append((acc/galleryTest.shape[2])*100)
                accRank=accRank+(acc/galleryTest.shape[2])*100

            if(accRank>maxAccRank):
                maxSigma=sigma
                maxAlpha=alpha
                maxAccRank=accRank
                bestAccArray=accArray
            print("Rank [1,3,5,10] accuracy with alpha=",alpha," and sigma=",sigma,":",accArray)
            

        print("best sigma:",maxSigma,"best alpha",maxAlpha,"Best rank [1,3,5,10] accuracy: ",bestAccArray)
                

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./prid_2011/single_shot/', type=str)
    parser.add_argument('--grid_step', default=4, type=int)
    parser.add_argument('--patchSize', default=10, type=int)
    parser.add_argument('--numberOfImageInGallery', default=10, type=int)

    parser.add_argument('--nbins', default=32, type=int)
    parser.add_argument("--scales",nargs="*",type=float,default=[1,0.75,0.5])
    args = parser.parse_args()

    main(args)

    

    
    
    
    
