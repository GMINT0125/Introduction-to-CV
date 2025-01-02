import numpy as np
import matplotlib.pyplot as plt
import cv2

def img_show(title, image):
    image = np.array(image, dtype=np.uint8)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def get_transformed_image (img,M):
    
    height, width = img.shape
    canvas = np.ones((801, 801), dtype=np.uint8) * 255
    result = np.ones((801, 801), dtype=np.uint8) * 255

    std = (400-height//2, 400-width//2) #시작점 좌표

    canvas[std[0]:std[0]+height, std[1]:std[1]+width] = img 


    #행 = I, 열 = J / 좌표로 생각하면 행 = y, 열 = x 순서가..
    # x = J, y = I
    # [400, 400]이 (0,0)이 되야하고 [0,0]은 (-400, 400)이 되어야 함.
    # x = J - 400, y = 400 - I

    for i in range(std[0], std[0]+height):            
        for j in range(std[1], std[1]+width):
            coord = np.array([j-400,400-i, 1]) #2차원 좌표로 변환..
            tf = np.dot(M, coord) #변환된 좌표를 2d transformation
            
            x, y = tf[0], tf[1] #x, y 좌표
            x, y = int(x+400), int(400-y)
            if 0 <= x < canvas.shape[1] and 0 <= y < canvas.shape[0]: #범위 바깥으로 나가면 무시
                result[y, x] = canvas[i, j]


    cv2.arrowedLine(result, (0, 400), (801, 400),(0, 0, 0), 2, tipLength=0.02)
    cv2.arrowedLine(result, (400, 801), (400, 0),(0, 0, 0), 2, tipLength=0.02)
    
    return result

def main():
    #1-1 Visualization of a transformed image on a 2D plane.

    print('1-1 Visualization of a transformed image on a 2D plane.')
    smile = cv2.imread('./CV_Assignment_2_Images/smile.png', cv2.IMREAD_GRAYSCALE) 

    canvas = get_transformed_image(smile, np.array([[1,0,0],[0,1,0],[0,0,1]]))
    img_show('canvas', canvas)

    #1-2 Interactive 2D transformations
    print('1-2 Interactive 2D transformations ')
    M = np.array([[1,0,0],[0,1,0],[0,0,1]])   #기본 M이고 입력받는 key에 따라 전환 될 것

    while True:

        canvas = get_transformed_image(smile, M)

        canvas = np.array(canvas)
        cv2.imshow('canvas', canvas)
        key = cv2.waitKey(0) 

        if key == ord('q'):
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break

        elif key == ord('w'): #Move to the upward by 5 pixels
            M = np.dot(np.array([[1,0,0],[0,1,5],[0,0,1]]), M)
        elif key == ord('s'): #Move to the downward by 5 pixels
            M = np.dot(np.array([[1,0,0],[0,1,-5],[0,0,1]]), M)
        elif key == ord('a'): #Move to the left by 5 pixels
            M = np.dot(np.array([[1,0,-5],[0,1,0],[0,0,1]]), M)
        elif key == ord('d'): #Move to the right by 5 pixels
            M = np.dot(np.array([[1,0,5],[0,1,0],[0,0,1]]), M)
        elif key == ord('f'): #Flip across  y  axis
            M = np.dot(np.array([[-1,0,0],[0,1,0],[0,0,1]]), M)
        elif key == ord('g'): #Flip across  x  axis
            M = np.dot(np.array([[1,0,0],[0,-1,0],[0,0,1]]), M)
        elif key == ord('r'): #Rotate counter-clockwise by 5 degrees 
            theta = np.deg2rad(5)
            M = np.dot([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]], M)
        elif key == ord('t'): #Rotate clockwise by 5 degrees
            theta = np.deg2rad(-5)
            M = np.dot([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]], M)
        elif key == ord('x'): #Shirnk the size by 5% along to  x  direction
            M = np.dot([[0.95,0,0],[0,1,0],[0,0,1]], M)
        elif key == ord('c'): #Enlarge the size by 5% along to  x  direction
            M = np.dot([[1.05,0,0],[0,1,0],[0,0,1]], M)
        elif key == ord('y'): #Shirnk the size by 5% along to  y  direction
            M = np.dot([[1,0,0],[0,0.95,0],[0,0,1]], M)
        elif key == ord('u'): #Enlarge the size by 5% along to  y  direction
            M = np.dot([[1,0,0],[0,1.05,0],[0,0,1]], M)

        elif key == ord('h'): #Restore to the initial state
            M = np.array([[1,0,0],[0,1,0],[0,0,1]])


if __name__ == '__main__':
    main()