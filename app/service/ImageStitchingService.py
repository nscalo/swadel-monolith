import cv2
import numpy as np

class ImageStitchingService():

    def __init__(self, imagePaths, size):
        self.imgs = []
        height, width = size
        for imagePath in imagePaths:
            image = cv2.imread(imagePath)
            self.imgs.append(cv2.resize(image, {height,width}))

    def perform_stitching(self, output_file):
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
        stitched = np.zeros((self.imgs[0].shape[0],self.imgs[0].shape[1]*len(self.imgs)))
        (status, stitched) = stitcher.stitch(self.imgs, stitched)
        if status == 0:
            # write the output stitched image to disk
            cv2.imwrite(output_file, stitched)
        
        return status, stitched
    
