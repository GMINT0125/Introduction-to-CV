import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings 
import time

warnings.filterwarnings("ignore")

def img_show(title, image):
    image = np.array(image, dtype=np.uint8)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

#2-2 compute homography

#2-2 homography

def compute_homography( srcP , destP ): #N x 2  matrix  (src x, src y), (dest x, dest y)

    #Normalization
    src_mean = np.mean(srcP, axis = 0)
    srcP = srcP - src_mean
    
    src_distance = np.sqrt(np.sum(srcP ** 2, axis = 1))

    src_max_distance = np.max(src_distance)
    src_scale = np.sqrt(2) / src_max_distance

    srcP = srcP * src_scale


    dest_mean = np.mean(destP, axis = 0)
    destP = destP - dest_mean 

    dest_distance = np.sqrt(np.sum(destP ** 2, axis = 1))

    dest_max_distance = np.max(dest_distance)
    dest_scale = np.sqrt(2) / dest_max_distance

    destP = destP * dest_scale

    s_1 = np.array([[1, 0, -src_mean[0]],[0, 1, -src_mean[1]], [0, 0, 1]])
    s_2 = np.array([[src_scale, 0, 0], [0, src_scale, 0], [0, 0, 1]])
    
    Ts = np.dot(s_2, s_1)

    d_1 = np.array([[1, 0, -dest_mean[0]],[0, 1, -dest_mean[1]], [0, 0, 1]])
    d_2 = np.array([[dest_scale, 0, 0], [0, dest_scale, 0], [0, 0, 1]])

    Td = np.dot(d_2, d_1)

    # Xd = H * Xs
    for i in range(len(srcP)): #i = index of coord
        src_x, src_y = srcP[i]
        dest_x, dest_y = destP[i]
        if i == 0:
            A = np.array([[-src_x, -src_y, -1, 0, 0, 0, src_x * dest_x, src_y * dest_x, dest_x],
                        [0, 0, 0, -src_x, -src_y, -1, src_x * dest_y, src_y * dest_y, dest_y]])
        else:
            A = np.concatenate((A, np.array([[-src_x, -src_y, -1, 0, 0, 0, src_x * dest_x, src_y * dest_x, dest_x],
                        [0, 0, 0, -src_x, -src_y, -1, src_x * dest_y, src_y * dest_y, dest_y]])), axis = 0)

    U, S, V = np.linalg.svd(A) #V = transpose V
     
    H = V[-1].reshape((3, 3))
    H = np.dot(np.dot(np.linalg.inv(Td), H), Ts)
    
    return H

# 2-3 RANSAC
def compute_homography_ransac( srcP , destP, th ):
    max_inliers = 0

    for i in range(3000):
        np.random.seed(i)
        inliers = []

        #potential matching 중 4개 골라서 H 구하기. 
        #검증 : 모든 Matching에 대해서 error 계산.
        #가장 많은 inlier들로 최종 H 구한다.

        index = np.random.choice(len(srcP), size=4, replace=False)
        temp_src = srcP[index]
        temp_dest = destP[index]
        current_H = compute_homography(temp_src, temp_dest)

        #validation
        for i in range(len(srcP)):
            
            src, dest = srcP[i], destP[i]
            src = np.append(src, 1)
            tfd = np.dot(current_H, src)
            tfd = tfd/tfd[2]
            tfd = tfd[:2]

            #compute error
            error = np.sqrt(np.sum((dest - tfd) ** 2))
            if error <= th:
                arr = np.array([srcP[i],destP[i]])
                inliers.append(arr)
        
        if len(inliers) >= max_inliers:
            max_inliers = len(inliers)
            inliers = np.array(inliers)
            src_inliers = inliers[:,0]
            dest_inliers = inliers[:,1]
            final_H = compute_homography(src_inliers, dest_inliers)

    
    return final_H

def main():
    #2-1
    desk = cv2.imread('CV_Assignment_2_Images/cv_desk.png', cv2.IMREAD_GRAYSCALE)
    cv_cover = cv2.imread('CV_Assignment_2_Images/cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
    hp_cover = cv2.imread('CV_Assignment_2_Images/hp_cover.jpg', cv2.IMREAD_GRAYSCALE)

    hp_cover = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

    orb = cv2.ORB_create()

    kp_desk = orb.detect(desk, None)
    kp_desk, des_desk = orb.compute(desk, kp_desk)

    kp_cover = orb.detect(cv_cover, None)
    kp_cover, des_cover = orb.compute(cv_cover, kp_cover)

    #compute hamming distace
    des = []
    for i in range(len(des_desk)):
        for j in range(len(des_cover)):
            dis = np.count_nonzero(des_desk[i] != des_cover[j])
            temp = (i, j , dis) #desk의 i / cover의 j번째 hamming distance
            des.append(temp)

    des = sorted(des, key = lambda x : x[2])

    des_ = []
    for i, j, dis in des:
        match = cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=dis)
        des_.append(match)

    result = cv2.drawMatches(desk, kp_desk, cv_cover, kp_cover, des_[:10], None, flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    img_show('Matching', result)

    #2-4
    #a) normalize

    desk_copy1 = desk.copy()
    srcpt = [] 
    destpt = []
    match = des_[:10] + des_[20:30] 
    for des in match:
        srcpt.append(kp_cover[des.trainIdx].pt) #(x,y) 형식
        destpt.append(kp_desk[des.queryIdx].pt)

    H = compute_homography(np.array(srcpt), np.array(destpt))
    normalized_converted = cv2.warpPerspective(cv_cover, H, (desk.shape[1], desk.shape[0]))

    #img_show('normalized', normalized_converted)

    for i in range(normalized_converted.shape[0]):
        for j in range(normalized_converted.shape[1]):
            if normalized_converted[i][j] != 0:
                desk_copy1[i][j] = normalized_converted[i][j]

    #img_show('normalized', desk_copy1)

    combined_normalized = np.hstack((normalized_converted, desk_copy1))
    img_show('Homography with normalization', combined_normalized)

    #b) RANSAC
    desk_copy2 = desk.copy()
    srcpt = [] 
    destpt = []
    match = des_[:15] + des_[20:40] + des_[90:100]#N should be larget than 15

    for des in match:
        srcpt.append(kp_cover[des.trainIdx].pt) #(x,y) 형식
        destpt.append(kp_desk[des.queryIdx].pt)

    start = time.time()
    H = compute_homography_ransac(np.array(srcpt), np.array(destpt), 10)
    end = time.time()
    con = end - start
    print('RANSAC TIME :', con)

    ransac_converted = cv2.warpPerspective(cv_cover, H, (desk.shape[1], desk.shape[0]))

    #img_show('RANSAC', ransac_converted)
    for i in range(ransac_converted.shape[0]):
        for j in range(ransac_converted.shape[1]):
            if ransac_converted[i][j] != 0:
                desk_copy2[i][j] = ransac_converted[i][j]

    #img_show('RANSAC', desk_copy2)

    combined_ransac = np.hstack((ransac_converted, desk_copy2))
    img_show('Homography with RANSAC', combined_ransac)

    #Nomalized v.s. RANSAC
    for_diff = np.vstack((combined_normalized, combined_ransac))
    img_show('Normalized vs RANSAC / 1st row : normalized / 2nd row: RANSAC', for_diff)

    #c)
    desk_copy3 = desk.copy()
    hp_converted = cv2.warpPerspective(hp_cover, H, (desk.shape[1], desk.shape[0]))

    for i in range(hp_converted.shape[0]):
        for j in range(hp_converted.shape[1]):
            if hp_converted[i][j] != 0:
                desk_copy3[i][j] = hp_converted[i][j]

    combined_hp = np.hstack((hp_converted, desk_copy3))
    img_show('harry potter with RANSAC', combined_hp)

if __name__ == '__main__':
    main()