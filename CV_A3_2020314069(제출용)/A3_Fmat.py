import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
random.seed(0)

from CV_A3_P1_Data.compute_avg_reproj_error import compute_avg_reproj_error

def img_show(title, image):
    image = np.array(image, dtype=np.uint8)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def compute_F_raw(M): #len(M) = matching point
    A = np.zeros((len(M),9))
    for idx, row in enumerate(M):
        x1, y1, x2, y2 = row
        row = [x1*x2, x2*y1, x2, x1*y2, y1*y2, y2, x1, y1, 1]
        A[idx] = row
    A = np.asarray(A)

    U, S ,V = np.linalg.svd(A)
    F = np.reshape(V[-1,:],(3,3))

    return F

def normalization(coord):
    mean = np.mean(coord, axis=0)
    coord = coord - mean

    distance = np.sqrt(np.sum(coord ** 2, axis=1))
    max_distance = np.max(distance)
    scale = np.sqrt(2) / max_distance
    coord = coord * scale

    T1 = np.array([[1, 0, -mean[0]], [0, 1, -mean[1]], [0, 0, 1]])
    T2 = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])

    T = np.dot(T2, T1)
    return T, coord

def rank2_constraint(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F_ = np.dot(U, np.dot(np.diag(S),V))
    return F_

def compute_F_norm(M):
    #normalize points
    #normalize and scale point by Matrix T
    x1y1 = M[:,0:2]
    x2y2 = M[:,2:4]
    
    T, x1y1 = normalization(x1y1)
    T_, x2y2 = normalization(x2y2)
    A = np.zeros((len(M),9))

    for idx in range(len(M)):
        x1, y1 = x1y1[idx]
        x2, y2 = x2y2[idx]
        row = [x1*x2, x2*y1, x2, x1*y2, y1*y2, y2, x1, y1, 1]
        A[idx] = row
    A = np.asarray(A)
    U, S ,V = np.linalg.svd(A)
    F = np.reshape(V[-1,:],(3,3))
    F = rank2_constraint(F)

    F = np.dot(T_.T, np.dot(F, T))

    return F

def compute_F_mine(M): 

    F = compute_F_norm(M)
    error = compute_avg_reproj_error(M, F)

    min_error = error
    best_F = F

    for i in range(1000):
        num_random_point = random.randint(8, len(M))
        random_point = random.sample(range(len(M)), num_random_point)
        ranM = M[random_point]
        ranF = compute_F_norm(ranM)
        ranerror = compute_avg_reproj_error(M, ranF)

        if ranerror < min_error:
            min_error = ranerror
            best_F = ranF

    return best_F


def visualize(image, F): #F는 이미 계산됨
    if image == 'temple':
        img1 = cv2.imread('./CV_A3_P1_Data/temple1.png', cv2.IMREAD_COLOR)
        img2 = cv2.imread('./CV_A3_P1_Data/temple2.png', cv2.IMREAD_COLOR)
    else:
        img1 = cv2.imread('./CV_A3_P1_Data/' + image + '1.jpg', cv2.IMREAD_COLOR)
        img2 = cv2.imread('./CV_A3_P1_Data/' + image + '2.jpg', cv2.IMREAD_COLOR)
    M = np.loadtxt('./CV_A3_P1_Data/' + image + '_matches.txt')

    idx = np.random.choice(len(M), 3) #3개 고르기
    coord = M[idx] #3x4 matrix
    p, q = coord[:,0:2], coord[:,2:4]

    RGB = [(0,0,255), (0,255,0), (255,0,0)]

    for i in range(3):
        x1, y1 = p[i]
        epi_line1 = np.dot(F, np.array([x1, y1, 1]))

        x2, y2 = q[i]
        epi_line2 = np.dot(F.T, np.array([x2, y2, 1]))
        #ax + by + c = 0 -> 다른 이미지에 대한 직선
        #start => x = 0, y = -c/b
        #end => x = img2.shape[1], y = -(c + a*img2.shape[1])/b

        start = (0, int(-epi_line1[2]/epi_line1[1]))
        end = (img2.shape[1], int(-(epi_line1[2] + epi_line1[0] * img2.shape[1])/epi_line1[1]))
        img2 = cv2.line(img2, start, end, RGB[i], 1)
        img2 = cv2.circle(img2, (int(x2), int(y2)), 5, RGB[i], -1)


        start = (0, int(-epi_line2[2]/epi_line2[1]))
        end = (img1.shape[1], int(-(epi_line2[2] + epi_line2[0] * img1.shape[1])/epi_line2[1]))
        img1 = cv2.line(img1, start, end, RGB[i], 1)
        img1 = cv2.circle(img1, (int(x1), int(y1)), 5, RGB[i], -1)

        result = np.hstack((img1, img2))
    return result    


def main():
    images = ['temple', 'house', 'library']
    for image in images:
        M = np.loadtxt('./CV_A3_P1_Data/' + image + '_matches.txt')
        F = compute_F_mine(M)

        raw_error = compute_avg_reproj_error(M,compute_F_raw(M))
        norm_error= compute_avg_reproj_error(M,compute_F_norm(M))
        mine_error= compute_avg_reproj_error(M,compute_F_mine(M))
        
        if image == 'temple':
            print('Average Reprojection Errors ({0} and {1})'.format(image+'1.png', image+'2.png'))
        else:
            print('Average Reprojection Errors ({0} and {1})'.format(image+'1.jpg', image+'2.jpg'))

        print('Raw  = ',raw_error)
        print('Norm = ', norm_error)
        print('Mine = ', mine_error,'\n')

        while True:
            result = visualize(image, F)
            result = np.array(result)

            cv2.imshow('result', result)

            key = cv2.waitKey(0)
            if key == ord('q'):
                cv2.destroyAllWindows()
                cv2.waitKey(1)
                break

if __name__ == '__main__':
    main()