#Canny Edge Detection Algorithm
import math
import sys
import  cv2
import numpy as np
import glob
import matplotlib.pyplot as plt


CROPRATE = 2
#gauss calculation
def gauss(x,y,sigma):
    normal = 1 / (2.0 * np.pi * sigma**2.0)
    exp_term = np.exp(-1*((x**2.0 + y**2.0) / (2.0 * sigma**2.0)))
    return (normal*exp_term)
#generating gauss filer matrix
def generate_gaussian_filter(size=5, sigma=0.375):
    gauss_kern = np.zeros((size,size))
    shalf = size // 2
    i=0
    for x in gauss_kern:
        j=0
        for y in x:
            gauss_kern[i,j] = gauss(i-shalf,j-shalf,2)
            j+=1
        i+=1

    return gauss_kern / np.sum(gauss_kern)

print(generate_gaussian_filter(5,0.375))
#apply gausian filter
def convolution (image, filter):

    if len(image.shape) == 3:
        m_i, n_i, c_i = image.shape

    
    elif len(image.shape) == 2:
        image = image[..., np.newaxis]
        
        m_i, n_i, c_i = image.shape
    else:
        raise Exception('Shape of image not supported')
    
    m_i, n_i, c_i, = image.shape

    m_k, n_k = filter.shape

    y_iter = m_i - m_k + 1 
    x_iter = n_i - n_k + 1

    img = image.copy()

    output_s = (y_iter,x_iter, c_i)
    output = np.zeros(output_s, dtype=np.float32)

    count = 0

    output_tmp = output.reshape((output_s[0]*output_s[1],output_s[2]))

    for i in range(y_iter):
        for j in range(x_iter):
            for z in range(c_i):
                submatrix = img[i:i+m_k,j:j+n_k,z]

                output_tmp[count , z] = np.sum(submatrix * filter)

            count +=1

    output = output_tmp.reshape(output_s)

    return output.astype(np.uint8)


#Non-Max-Suppression function
def non_maximum_suppression(image, angles):
    size = image.shape
    suppressed = np.zeros(size)
    for i in range(1, size[0] - 1):
        for j in range(1, size[1] - 1):
            x = 255 
            y = 255
            if (0 <= angles[i, j] < 22.5) or (157.5 <= angles[i, j] <= 180):
                x = image[i,j-1]
                y = image[i,j+1]
            elif (22.5 <= angles[i, j] < 67.5):
                x = image[i-1,j-1]
                y = image[i+1,j+1]
            elif (67.5 <= angles[i, j] < 112.5):
                x = image[i-1,j]
                y = image[i+1,j]
            else:
                x = image[i+1,j-1]
                y = image[i-1,j+1]
            
            if (image[i, j] >= x) and (image[i,j] >= y):
                suppressed[i, j] = image[i, j]
            else:
                suppressed[i,j] = 0
    suppressed = np.multiply(suppressed, 255.0 / suppressed.max())
    return suppressed.astype(np.uint8)

#Double Thresholding function 
def double_thresholding(image,low,high):
    dt_image = np.zeros(image.shape)
    for i in range(0,dt_image.shape[0]):
        for j in range(0,dt_image.shape[1]):
            if image[i,j] < low:
                dt_image[i,j] = 0
            elif image[i,j] >= low and image[i,j] < high:
                dt_image[i,j] = 128
            else:
                dt_image[i,j] = 255

    return dt_image.astype(np.uint8)

#Hysteresis function 
def hysteresis(thresholded):
    strong = np.zeros(thresholded.shape)
    
    for i in range(0,thresholded.shape[0]):
        for j in range(0,thresholded.shape[1]):
            val = thresholded[i,j]

            
            if val == 128:
                
                if thresholded[i-1,j] == 255  or thresholded[i-1,j-1] == 255 or thresholded[i-1,j+1] == 255 or thresholded[i+1,j-1] == 255 or thresholded[i+1,j+1] == 255 or thresholded[i+1,j] == 255 or thresholded[i,j-1] == 255 or thresholded[i,j+1] == 255:
                        strong[i,j] = 255

            elif val == 255:
                strong[i,j] = 255
            
    return strong.astype(np.uint8)


#calculate threshold values
def cal_threshold(img):
    v = np.mean(img)

    l = int(0.1 * v)

    u = int(0.9 * v)
    return (u,l)
    
#Hist funciton for get the histagrom of grayscale image
def Hist(img):
   row, col = img.shape 
   y = np.zeros(256)
   for i in range(0,row):
      for j in range(0,col):
         y[img[i,j]] += 1
   x = np.arange(0,256)
   plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
   plt.show()
   return y


#Canny function or applying all steps and functions
def Canny_detector(img):
    #downscale image for faster proccessing
    height, width , x = img.shape
    width = round(width/CROPRATE)
    height = round(height/CROPRATE)
    img = cv2.resize(img,(width,height))

    #convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    #apply gaussian filter
    blurimage = convolution(gray, generate_gaussian_filter(3,0.6))

    #apply sobel filter
    grad_x = cv2.Sobel(blurimage,cv2.CV_64F,1,0,ksize=3)
    grad_y = cv2.Sobel(blurimage,cv2.CV_64F,0,1,ksize=3)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    #combine x and y dimension of grandiant
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    #calculate angels
    angles = np.rad2deg(np.arctan2(grad_y, grad_x))

    #non-max-suppression
    nms = non_maximum_suppression(grad , angles)

    #histogram of grayscale image
    hist = Hist(grad)

    #high,low =cal_threshold(grad)
    high = 40 
    low = high *0.4

    #print threshold values
    print("threshold values --> high :", high, "low: ",low)

    #apply double threshold to image
    dt_image = double_thresholding(nms , low, high)

    #apply hysteresis
    strong = hysteresis(dt_image)
    
    #put all steps to an array to show
    imgs = [gray,blurimage,grad_x,grad_y,grad,nms,dt_image,strong]
    
    maxX = imgs[0].shape[0]
    maxY = imgs[0].shape[1]
    for image in imgs:
        if image.shape[0] > maxX:
            maxX = image.shape[0]
        if image.shape[1] > maxY:
            maxY = image.shape[1]

    print("maxX:",maxX," maxY:",maxY)
    
    
    #fix the all images sizes
    for i in range(0,len(imgs)):

        if imgs[i].shape[0] != maxX and imgs[i].shape[1] != maxY:
            borderX = int((maxX - imgs[i].shape[0])/2)
            borderY = int((maxY - imgs[i].shape[1])/2)
            imgs[i] = cv2.copyMakeBorder(imgs[i], borderX, borderX, borderY, borderY, cv2.BORDER_CONSTANT,value=0)

        
        if len(imgs[i].shape) > 2:
            
            if imgs[i].shape[2] == 3:
                continue
                
        imgs[i] = (np.dstack((imgs[i],imgs[i],imgs[i]))).astype(np.uint8) 
    res1 = np.concatenate(imgs,1)

    return res1 


img = cv2.imread("C:\\Users\\asm\Desktop\\MERT AUTO FOCUS\\asd.jpg")

image = Canny_detector(img)


cv2.imshow('gray-blurred-gradx-grady-grad-nms-thresholded-hysteresis', image)

cv2.waitKey(0)
 
cv2.destroyAllWindows()


    

    