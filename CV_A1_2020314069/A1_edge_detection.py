from A1_image_filtering import *
import time
import cv2
import warnings

warnings.filterwarnings("ignore")

#2-2  Implement a function that returns the image gradient
def compute_image_gradient(img):
    S_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    S_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    grad_x = cross_correlation_2d(img, S_x)
    grad_y = cross_correlation_2d(img, S_y)

    mag = np.sqrt(grad_x**2+grad_y**2)
    dir = np.arctan2(grad_y, grad_x)

    return mag, dir


#2-3  Implement a function that performs Non-maximum Suppression (NMS)
def quantize_dir(dir):
    quantized_dir = np.zeros_like(dir) #dir.shape == image.shape -> quantized_dir
    degree = np.degrees(dir) % 360 
    angle = [0, 45, 90, 135, 180, 225, 270, 315, 360]

    for i in range(degree.shape[0]):
        for j in range(degree.shape[1]):

            nearest_angle_idx =  np.argmin(np.abs([temp - degree[i][j] for temp in angle]))
            nearest_angle = angle[nearest_angle_idx]
            quantized_dir[i][j] = nearest_angle
            
    quantized_dir[quantized_dir == 360] = 0

    return quantized_dir

def non_maximum_suppression_dir(mag, dir):
    dir = quantize_dir(dir)
    result = np.zeros_like(mag)
    #(0, 180) -> 좌우 픽셀 비교 -> ([0, 1], [0, -1])
    #(45, 225) -> 대각선 픽셀 비교 -> ([1, -1], [-1, 1])
    #(90, 270) -> 상하 픽셀 비교 -> ([1, 0], [-1, 0])
    #(135, 315) -> 대각선 픽셀 비교 -> ([-1, -1], [1, 1])
    #query가 되는 픽셀이 끝에 있는 경우 하나의 픽셀만 비교하면 된다.
    for i in range(dir.shape[0]):
        for j in range(dir.shape[1]):

            angle = dir[i][j]
            mag_value = mag[i][j]

            if angle == 0 or angle == 180:
                if j == 0:
                    if mag_value >= mag[i][j+1]:
                        result[i][j] = mag_value        
                elif j == dir.shape[1]-1:
                    if mag_value >= mag[i][j-1]:
                        result[i][j] = mag_value
                else:
                    if mag_value >= mag[i][j-1] and mag_value >= mag[i][j+1]:
                        result[i][j] = mag_value 

            elif angle == 90 or angle == 270:
                if i == 0:
                    if mag_value >= mag[i+1][j]:
                        result[i][j] = mag_value
                elif i == dir.shape[0]-1:
                    if mag_value >= mag[i-1][j]:
                        result[i][j] = mag_value
                else:
                    if mag_value >= mag[i-1][j] and mag_value >= mag[i+1][j]:
                        result[i][j] = mag_value


            elif angle == 45 or angle == 225:
                #극값인 경우
                if (i == 0 and j==0) or (i == dir.shape[0]-1 and j == dir.shape[1]-1):
                    result[i][j] = mag_value
                #첫번째 행의 경우
                elif i == 0:
                    if mag_value >= mag[i+1][j-1]:
                        result[i][j] = mag_value
                #마지막 행의 경우
                elif i == dir.shape[0]-1:
                    if mag_value >= mag[i-1][j+1]:
                        result[i][j] = mag_value
                #첫번째 열의 경우
                elif j == 0:
                    if mag_value >= mag[i-1][j+1]:
                        result[i][j] = mag_value
                #마지막 열의 경우
                elif j == dir.shape[1]-1:
                    if mag_value >= mag[i+1][j-1]:
                        result[i][j] = mag_value

                else:
                    if mag_value >= mag[i-1][j+1] and mag_value >= mag[i+1][j-1]:
                        result[i][j] = mag_value



            elif angle == 135 or angle == 315:
                #극값인 경우
                if( i == 0 and j == dir.shape[1]-1) or (i == dir.shape[0]-1 and j == 0):
                    result[i][j] = mag_value

                #첫번째 행의 경우
                elif i == 0:
                    if mag_value >= mag[i+1][j+1]:
                        result[i][j] = mag_value
                #마지막 행의 경우
                elif i == dir.shape[0]-1:
                    if mag_value >= mag[i-1][j-1]:
                        result[i][j] = mag_value

                #첫번째 열의 경우
                elif j == 0:
                    if mag_value >= mag[i+1][j+1]:
                        result[i][j] = mag_value

                #마지막 열의 경우
                elif j == dir.shape[1]-1:
                    if mag_value >= mag[i-1][j-1]:
                        result[i][j] = mag_value
                else:
                    if mag_value >= mag[i-1][j-1] and mag_value >= mag[i+1][j+1]:
                        result[i][j] = mag_value


    return result

def main():
    images = ['lenna.png', 'shapes.png']
    kernel = get_gaussian_filter_2d(7, 1.5)
    results = []

    #2-1 apply the gaussian filtering to the input image
    for image_name in images:
        image = cv2.imread('A1_Images/'+ image_name, cv2.IMREAD_GRAYSCALE)
        result= cross_correlation_2d(image, kernel)
        results.append(result)

    #2-2
    for i in range(len(results)):
        image = results[i]
        image_name = images[i]

        start = time.time()
        mag, dir = compute_image_gradient(image)
        end = time.time()
        consumption = end - start
        print('Time consumption of compute image gradient for {} : {}'.format(image_name, consumption))
        img_show('Computed Magnitude Map for {}'.format(image_name), mag)
        cv2.imwrite('./result/part_2_edge_raw_' + image_name, mag)
    print('')
    
    #2-3
    for i in range(len(results)):
        image = results[i]
        image_name = images[i]
        mag, dir = compute_image_gradient(image)
        start = time.time()
        result = non_maximum_suppression_dir(mag, dir)
        end = time.time()
        print('Time consumption of non maximum suppression for {} : {}'.format(image_name, end-start))
        #show the supressed manitude map and store it to an image file
        img_show('Supressed Magnitude Map for {}'.format(image_name), result)
        cv2.imwrite('./result/part_2_edge_sup_' + image_name, result)
    

if __name__ == '__main__':
    main()