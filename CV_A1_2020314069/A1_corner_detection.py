from A1_image_filtering import *
from A1_edge_detection import *

import numpy as np
import time
import cv2

import warnings
warnings.filterwarnings("ignore")

def zero_pad(image, size):
    height, width = image.shape[0], image.shape[1]
    padded_image = np.zeros((height+2*size, width+2*size))
    padded_image[size:size+height, size:size+width] = image
    return padded_image


def compute_derivatives(img):
    S_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    S_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    grad_x = cross_correlation_2d(img, S_x)
    grad_y = cross_correlation_2d(img, S_y)
    
    return grad_x, grad_y


def compute_corner_response(img):
    k = 0.04

    height, width = img.shape[0], img.shape[1] 
    grad_x, grad_y = compute_derivatives(img) 
    padded_grad_x = zero_pad(grad_x,2) 
    padded_grad_y = zero_pad(grad_y,2)
    result = np.zeros((height, width)) 

    for i in range(0, height): 
        for j in range(0, width):            
            
            sampled_grad_x = padded_grad_x[i:i+5, j:j+5]
            sampled_grad_y = padded_grad_y[i:i+5, j:j+5]

            I_x = np.sum(sampled_grad_x**2)
            I_y = np.sum(sampled_grad_y**2)
            I_xy = np.sum(sampled_grad_x * sampled_grad_y)

            matrix = np.array([[I_x, I_xy],[I_xy, I_y]])

            R = np.linalg.det(matrix) - k*(np.trace(matrix)**2)

            result[i, j] =  R
    
    result[result<0] = 0
    result = result/np.max(result)
    return result


def non_maximum_suppression_win(R, winSize):
    sup_R = np.zeros_like(R)
    padded_R = zero_pad(R, winSize//2)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            sampled_R = padded_R[i:i+winSize, j:j+winSize]
            maximum_value = np.max(sampled_R)
            if R[i,j] != maximum_value:
                sup_R[i,j] = 0
            else: 
                sup_R[i,j] = maximum_value
    sup_R[sup_R<0.1] = 0
    return sup_R




def main():

#3-1 Applying Gaussian Filtering 
    images = ['lenna.png', 'shapes.png']
    kernel = get_gaussian_filter_2d(7, 1.5)

    origin = [] #Original Image
    results = [] #After Gaussian Filtering

    for image_name in images:
        image = cv2.imread('A1_Images/'+ image_name, cv2.IMREAD_GRAYSCALE)
        origin.append(image)
        result= cross_correlation_2d(image, kernel)
        results.append(result)

#3-2 Implement a function that returns corner response values

    Rs = [] #R을 저장할 리스트

    for i in range(len(images)):
        image_name = images[i]
        img = results[i] #Gaussian Filtering 된 이미지를 사용
        start = time.time()
        result = compute_corner_response(img)
        end = time.time()

        consumption = end - start

        Rs.append(result)
        result = result*255
        print("Consumption Time of compute Corner Response for {}: {}".format(image_name, consumption))
        img_show('Corner Response for {}'.format(image_name), result) 
        cv2.imwrite('./result/part_3_corner_raw_'+image_name, result)
    print("")

#3-3 (b)

    for i in range(len(images)):
        R = Rs[i]
        image_name = images[i]
        bgr_image = cv2.cvtColor(origin[i], cv2.COLOR_GRAY2BGR)
        R[R<0.1] = 0
        for j in range(bgr_image.shape[0]):
            for k in range(bgr_image.shape[1]):
                if R[j,k] > 0:
                    bgr_image[j,k] = [0,255,0]
        
        img_show('Corner Bin for {}'.format(image_name), bgr_image)
        cv2.imwrite('./result/part_3_corner_bin_'+image_name, bgr_image)

#3-3 (c) Implement a function that compute local maximas by non-maximum suppression

    for i in range(len(images)):
        R = Rs[i]

        start = time.time()
        sup_R = non_maximum_suppression_win(R, 11)
        end = time.time()
        consumption = end - start

        bgr_image = cv2.cvtColor(origin[i], cv2.COLOR_GRAY2BGR)
        for j in range(bgr_image.shape[0]):
            for k in range(bgr_image.shape[1]):
                if sup_R[j,k] > 0:
                    cv2.circle(bgr_image, (k, j), radius=3, color=(0, 255, 0), thickness=2)

        print("Consumption Time of Non Maximum Suppression for {}: {}".format(images[i], consumption))
        img_show('Corner Suppressed for {}'.format(images[i]), bgr_image)
        cv2.imwrite('./result/part_3_corner_sup_'+images[i], bgr_image)

    print('')
if __name__ == "__main__":
    main()