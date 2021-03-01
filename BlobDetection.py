import numpy as np
import PIL as p
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt



class OFilter:
    def __init__(self, order, mask_size):
        self.order = order
        self.mask_size = mask_size
        
    def local_filter(self, x):
        x.sort()
        return x[self.order-1]

    def ordfilt2(self, A):
        return ndimage.generic_filter(A, self.local_filter, size=(self.mask_size, self.mask_size))


def createScaleSpace(img,numScale,sigma,mul):
    h,w = img.shape
    scale_space = np.zeros((h,w,numScale))
    
    for i in range(1,numScale+1):
        scaled_sigma = sigma*(mul**(i))
        log = ndimage.gaussian_laplace(img,scaled_sigma)
        filtered_img = np.square((scaled_sigma**2)*log)
        scale_space[:,:,i-1] = filtered_img
    return scale_space
        

def detectBlobs(img,numScale,sigma,mul,threshold):
    scale_space = createScaleSpace(img,numScale,sigma,mul)
    h,w = img.shape
        
    #2D non-maximum supression
    d2_scalespace = np.zeros((h,w,numScale))
    for i in range(numScale):
        d2_scalespace[:,:,i] = nms_2D(scale_space[:,:,i],1)
    #3D Non-maximum supression   
    d3_scalespace = nms_3D(d2_scalespace,scale_space,numScale)
    
    return np.where(d3_scalespace > threshold,d3_scalespace,0) #return the value which are present in original scale space
    

def nms_3D(scalespacenms_2d,scale_space,numScale): #3d Non-maximum supression
    h,w = scalespacenms_2d[:,:,1].shape
    maxval_in_nbgScalespace = scalespacenms_2d
    for i in range(1,numScale-1):
        maxval_in_nbgScalespace[:,:,i] = np.max(maxval_in_nbgScalespace[:,:,:],axis = 2)
        
    return np.where(maxval_in_nbgScalespace==scale_space,maxval_in_nbgScalespace,0)


def nms_2D(img,radius):   #2d non maximum supression
    nbd_size = 2*radius+1 #mask size
    obj = OFilter(nbd_size**2,3)
    filtered_img = obj.ordfilt2(img)
    return filtered_img
    


def calcRadiusByScale(numScale,multiplier,sigma):
    rad = np.zeros((1,numScale))
    for i in range(numScale):
        rad[0][i] = np.sqrt(2)*sigma*(multiplier**i)
    return rad
    
def getBlobMarkers(d3_scale_space,rad_scale):
    h,w,numScale = d3_scale_space.shape
    cluster = []
    for i in range(numScale):
        row,col = np.where(d3_scale_space[:,:,i] != 0)
        radius = rad_scale[i]
        al = [row,col,radius]
        cluster.append(al)
    return cluster

def plot_blob_markers(blob_markers,img):
    fig, ax = plt.subplots(figsize =(12,10))
    ax.imshow(img,cmap ='gray')

    for blob in blob_markers:
        y,x,r = blob[0],blob[1],blob[2]
        for i in range(len(x)):
            c = plt.Circle((x[i], y[i]), r, color='red', linewidth=1.5, fill=False)

            ax.add_patch(c)
    ax.plot()
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_xlim(0,img.shape[1])
    ax.set_ylim(img.shape[0],0)
    plt.imsave("einstein_op.png",img)
    plt.show()

if __name__ == '__main__':
    numScale = 15
    sigma = 1
    multiplier = np.sqrt(np.sqrt(2))
    threshold = 0.010 # for binary image

    #read the image
    in_img = p.Image.open("../Inputs_BlobDetection/Inputs_BlobDetection/butterfly.jpg").convert('L')
    a = np.asarray(in_img)
    img = a/255 #convert to binary value

    d3_scale_space = detectBlobs(img,numScale,sigma,multiplier,threshold)
    
    rad_scale = calcRadiusByScale(numScale,multiplier,sigma)[0] #3d array containing blob markers and its corresponding cordinates pos.

    blob_markers = getBlobMarkers(d3_scale_space,rad_scale)

    plot_blob_markers(blob_markers,img)