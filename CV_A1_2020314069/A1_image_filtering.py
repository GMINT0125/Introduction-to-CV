import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

def img_show(title, image):
    image = np.array(image, dtype=np.uint8)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def img_pad(image, padding_size, axis):
    height, width = image.shape
    if axis == 0: #이미지 hegith 방향으로 padding
        pad_image = np.zeros((height+(2*padding_size), width)) # padding size 만큼 이미지의 height 방향으로 늘림
        pad_image[padding_size:padding_size+height, :] = image
        pad_image[:padding_size, :] = image[0, :]
        pad_image[padding_size+height:, :] = image[-1, :]

    elif axis == 1:
        pad_image = np.zeros((height, width+(2*padding_size))) # 가로 방향 Padding
        pad_image = np.transpose(pad_image) 
        image = np.transpose(image)

        pad_image[padding_size:padding_size+width, :] = image
        pad_image[:padding_size, :] = image[0, :]
        pad_image[padding_size+width:, :] = image[-1, :]

        pad_image = np.transpose(pad_image)

    return pad_image

def cross_correlation_1d(img, kernel): #가로 모양의 필터와 세로 모양의 필터 존재함을 고려함. kernel의 형태 (1,n) 또는 (n,1)이다.
    result = np.zeros((img.shape[0], img.shape[1])) #결과 이미지

    if kernel.shape[0] == 1: #(1,n) 커널 가로 방향 padding 필요
        padding_size = (kernel.shape[1] - 1) // 2
        padded_image = img_pad(img, padding_size, axis=1) #가로 방향 padding / padding된 이미지를 돌면서 filtering 해서 result에 저장

        for i in range(result.shape[0]):
            for j in range(result.shape[1]): #Reult 이미지의 각 픽셀에 대해
                result[i][j] = np.sum(padded_image[i, j:j+kernel.shape[1]] * kernel[0, :]) 


        
    elif kernel.shape[1] == 1: #(n,1) 
        padding_size = (kernel.shape[0] - 1) // 2
        padded_image = img_pad(img, padding_size, axis=0)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]): 
                result[i, j] = np.sum(padded_image[i:i + kernel.shape[0], j] * kernel[:, 0])

    return result

def cross_correlation_2d(img,kernel):
    result = np.zeros((img.shape[0], img.shape[1]))

    padding_size_x = (kernel.shape[1] - 1) // 2 #가로방향 padding axis = 1
    padding_size_y = (kernel.shape[0] - 1) // 2 #세로방향 padding axis = 0

    padded_image = img_pad(img, padding_size_y, axis=0)
    padded_image = img_pad(padded_image, padding_size_x, axis=1)

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i][j] = np.sum(padded_image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    
    return result


def get_gaussian_filter_1d(size, sigma): #1차원 가우시안 필터 가로냐 세로냐는 transpose로 해결하면 됨.. 중앙을 기준으로 대칭임.
    kernel = np.zeros((1, size)) #(1, 5) -> [1,2,3,4,5]
    center = size // 2 #center index 5의 경우 2

    for i in range(center+1): # i = 0, 1, 2
        gaussian_value = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(i**2)/(2*sigma**2))
        kernel[0][center-i] = gaussian_value
        kernel[0][center+i] = gaussian_value
    kernel= kernel/np.sum(kernel)
        
    return kernel

def get_gaussian_filter_2d(size, sigma): #2차원 가우시안 필터 center를 기준으로 대칭
    kernel = np.zeros((size, size))
    center = size // 2 #kernel[center][center] -> 중심이 되는값
    for i in range(center+1):
        for j in range(center+1):
            gaussian_value = 1/(2*np.pi*sigma**2) * np.exp(-((i**2+j**2)/(2*sigma**2)))
            kernel[center-i][center-j] = gaussian_value
            kernel[center-i][center+j] = gaussian_value
            kernel[center+i][center-j] = gaussian_value
            kernel[center+i][center+j] = gaussian_value
    kernel = kernel/np.sum(kernel)
    return kernel    

def main():

#   (c)
    kernel_1d =  get_gaussian_filter_1d(5, 1)
    kernel_2d = get_gaussian_filter_2d(5, 1)

    print('KERNEL_1D: \n{}'.format(kernel_1d))
    print('')
    print('KERNEL_2D: \n{}'.format(kernel_2d))  
    print('')


    images = ['lenna.png', 'shapes.png']

    position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    font_scale = 1 
    font_color = (0, 0, 0)
    thickness = 2 
    line_type = cv2.LINE_AA  

#   (d)
    for image_name in images:
        image = cv2.imread('A1_Images/'+image_name, cv2.IMREAD_GRAYSCALE)

        #Image Combination
        column = []
        for size in range(5,18,6):
            row = []
            for sigma in range(1, 12, 5):
                title = ("{}x{} s={}".format(size, size, sigma))
                kernel = get_gaussian_filter_2d(size, sigma)
                result = cross_correlation_2d(image, kernel)
                cv2.putText(result, title, position, font, font_scale, font_color, thickness, line_type)
                row.append(result)
            images_row = np.hstack(row)
            column.append(images_row)
        images = np.vstack(column)

        #image show & save
        img_show('Gaussian Filtered Image', images)
        cv2.imwrite('./result/part_1_gaussian_filtered_'+image_name, images)

#   (e)

    kernel_1d_x = get_gaussian_filter_1d(17, 6)
    kernel_1d_y = np.transpose(get_gaussian_filter_1d(17, 6)) 

    kernel_2d = get_gaussian_filter_2d(17, 6)
    
    images = ['lenna.png', 'shapes.png']

    for image_name in images:
        image = cv2.imread('A1_Images/'+image_name, cv2.IMREAD_GRAYSCALE)
        start_time = time.time()
        result1 = cross_correlation_1d(image, kernel_1d_y)
        result1 = cross_correlation_1d(result1, kernel_1d_x)
        end_time = time.time()
        consumption = end_time - start_time
        print('Sequential 1D gaussian consumption time for {} : {} \n'.format(image_name, consumption))
        img_show('Sequential 1d gaussian filtered image', result1)

    
        start_time = time.time()
        result2 = cross_correlation_2d(image, kernel_2d)
        end_time = time.time()

        consumption = end_time - start_time
        difference = np.abs(result2 - result1)

        print('2D gaussian consumption time for {} : {} \n'.format(image_name, consumption))
        img_show('2d gaussian filtered image', result2)

        img_show('Difference between 1D and 2D', difference)
        print('The sum of (absolute) intensity differences for {} : {} \n'.format(image_name, np.sum(difference)))
        print('')


if __name__ == '__main__':
    main()