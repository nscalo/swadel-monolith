import cv2
import numpy as np
from sklearn.neighbors import KDTree

class PlotFeatureExtractionService():

    scale = 0.1
    blockSize = 2
    apertureSize = 3
    k = 0.04
    neighbors = 3
    query_neighbors = 2
    kernerSize = 3
    filterSigma = 0.3
    
    def __init__(self):
        pass

    # set plot initial coordinates
    def set_data(self, height, width, scale=0.1):
        height = int(height / scale)
        width = int(height / scale)
        plot_data = np.arange(0,height*width,1)
        plot = np.zeros((height*width, 2))
        rows = plot_data / width
        cols = plot_data % width
        plot[:,1] = rows
        plot[:,0] = cols
        
        return plot * scale

    def pick_centres(self, src_gray, blurThreshold):
        blur = cv2.GaussianBlur(src_gray, PlotFeatureExtractionService.kernerSize, 
        sigmaX=PlotFeatureExtractionService.filterSigma, sigmaY=PlotFeatureExtractionService.filterSigma)
        
        return np.stack(np.where(blur[blur > blurThreshold]), axis=0)

    def set_feature_matching(self, src_gray, harrisThreshold, gaussianThreshold):
        # Detector parameters
        dst = cv2.cornerHarris(src_gray, 
        PlotFeatureExtractionService.blockSize, 
        PlotFeatureExtractionService.apertureSize, PlotFeatureExtractionService.k)
        dst_norm = np.empty(dst.shape, dtype=np.float32)
        dst_norm = cv2.normalize(dst, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        dst_idxs = np.where(dst_norm > harrisThreshold)
        points_map = np.stack(dst_idxs, axis=0)
        tree = KDTree(points_map, leaf_size=PlotFeatureExtractionService.neighbors)
        centres = self.pick_centres(src_gray, gaussianThreshold)
        dist, idx = tree.query(centres, k=PlotFeatureExtractionService.query_neighbors, return_distance=True)

        return points_map[idx]
    
    # iterate and do image stitching
    def __iter__(self):
        pass

    