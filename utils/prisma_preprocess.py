import rasterio
import numpy as np
from scipy.ndimage import rotate
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

def prisma_preproccessing(path: str) -> np.array:
    # Open the raster file and rotate it
    numpy_array = rasterio.open(path).read()
    print("the initial image has shape {}".format(numpy_array.shape))
    bands = numpy_array.shape[0]
    img_rotate = np.array([rotate(numpy_array[i,:,:], angle = 15.00, reshape= False, order =0) for i in range(bands)])
    img_rotate = img_rotate[:, 150:1080, 150:1080]
    # cut the absorbed bands
    print("cutting the absorbed bands...")
    final_list = []
    for i in range(bands):
        if (np.max(img_rotate[i,:,:] - np.min(img_rotate[i,:,:]) > 0.035)):
            final_list.append(img_rotate[i,:,:])
    img_final = np.array(final_list)
    print("the final image has shape {}".format(img_final.shape))
    print("the final image has max value {}".format(np.max(img_final)))
    print("the final image has min value {}".format(np.min(img_final)))
    return img_final

def prisma_svd(img_final: np.array, n_components: int, plot: bool = False, plot_variance: bool = False):
    # reshape the image
    rearrange = np.transpose(img_final, (1,2,0))
    print("the rearranged image has shape {}".format(rearrange.shape))
    img_1_reshaped = np.reshape(rearrange, (rearrange.shape[0]*rearrange.shape[1], rearrange.shape[2])) 
    print("the reshaped image has shape {}".format(img_1_reshaped.shape))
    # SVD decomposition
    svd = TruncatedSVD(n_components=n_components, n_iter=7, random_state=42)
    svd.fit(img_1_reshaped)
    transformed = svd.transform(img_1_reshaped)
    #transformed = min_max_normalization(transformed)
    #scaler = StandardScaler()
    #transformed = scaler.fit_transform(transformed)
    print("the transformed image has max value {}".format(np.max(transformed)))
    print("the transformed image has min value {}".format(np.min(transformed)))
    print("the transformed image has shape {}".format(transformed.shape))
    svd_img = np.reshape(transformed, (rearrange.shape[0], rearrange.shape[1], n_components))
    svd_img = np.transpose(svd_img, (2,0,1))
    print("the reshaped transformed image has shape {}".format(svd_img.shape))
    if plot:
        for i in range(n_components):
            im =plt.imshow(svd_img[i,:,:])
            plt.colorbar(im)
            plt.show()
        #plt.imshow(svd_img[0,:,:])
        #plt.show()
    # compute the explained variance
    explained_variance = svd.explained_variance_ratio_
    print("the explained variance is {}".format(explained_variance))
    if plot_variance:
        plt.plot(np.arange(1, n_components+1), explained_variance, 'o')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        plt.show()
    return svd_img

def min_max_normalization(img: np.array) -> np.array:
    # norm from 0 to 1
    scaler = MinMaxScaler()
    scaler.fit(img)
    img_norm = scaler.transform(img)
    print("the normalized image has max value {}".format(np.max(img_norm)))
    print("the normalized image has min value {}".format(np.min(img_norm)))
    return img_norm

def dataset_creation(img: np.array, patch_size: int, values: int, step =125) -> np.array:
    # create patches
    patches = patchify(img, (values, patch_size, patch_size), step=step)
    patches = np.reshape(patches, (patches.shape[0]*patches.shape[1]*patches.shape[2], patches.shape[3], patches.shape[4], patches.shape[5]))
    print("the shape is ",patches.shape)
    return patches

def gt_preproccessing(path: str) -> np.array:
    # Open the raster file and rotate it
    numpy_array = rasterio.open(path).read()
    print("the initial gt has shape {}".format(numpy_array.shape))
    bands = numpy_array.shape[0]
    img_rotate = np.array([rotate(numpy_array[i,:,:], angle = 15.00, reshape= False, order =0) for i in range(bands)])
    img_rotate = img_rotate[:, 150:1080, 150:1080]
    # cut the absorbed bands
    print("the final gt has shape {}".format(img_rotate.shape))
    print("the final gt has  value {}".format(np.unique(img_rotate)))
    return img_rotate

def basalt_patches(dataset, gt_dataset):
    basalt_patches = []
    basalt_gt = []
    for i in range(len(gt_dataset)):
        if 1 in gt_dataset[i, :, :, :]:
            basalt_patches.append(dataset[i, :, :, :])
            basalt_gt.append(gt_dataset[i, :, :, :])
    return np.array(basalt_patches), np.array(basalt_gt)

def moon_preprocess(path):
    numpy_array = rasterio.open(path).read()
    print("the initial image has shape {}".format(numpy_array.shape))
    #numpy_array = np.where(numpy_array < 0, 0, numpy_array)
    print("the initial image has max value {}".format(np.max(numpy_array)))
    print("the initial image has min value {}".format(np.min(numpy_array)))
    #bands = numpy_array.shape[0]
    #print("the initial image has max value {}".format(np.max(numpy_array)))
    #print("the initial image has min value {}".format(np.min(numpy_array)))
    #rearrange = np.transpose(numpy_array, (1,2,0))
    #img_1_reshaped = np.reshape(rearrange, (rearrange.shape[0]*rearrange.shape[1], rearrange.shape[2])) 
    #scaler = MinMaxScaler()
    #scaler.fit(img_1_reshaped)
    #img_norm = scaler.transform(img_1_reshaped)
    #print("the normalized image has max value {}".format(np.max(img_norm)))
    #print("the normalized image has min value {}".format(np.min(img_norm)))
    #norm_img = np.reshape(img_norm, (rearrange.shape[0], rearrange.shape[1], rearrange.shape[2]))
    #norm_img = np.transpose(norm_img, (2,0,1))
    #print("the reshaped transformed image has shape {}".format(norm_img.shape))
    return numpy_array
    



