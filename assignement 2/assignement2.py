'''
Assignement 02: Object counting
Group: <7>
Names: <Martens, Jonathan; Leisinger, Oliver>
Date: <12.11.2021>
Sources: <gdv template, tutorial 7, https://stackoverflow.com/questions/32522989/opencv-better-detection-of-red-color>
'''

import cv2
import glob # for loading all images from a directory
import numpy as np

### Goal: Count the number of all colored balls in the images

# ground truth
num_yellow = 30
num_blue = 5
num_pink = 8
num_white = 10
num_green = 2
num_red = 6
gt_list = (num_red, num_green, num_blue, num_yellow, num_white, num_pink)

# define color ranges in HSV, note that OpenCV uses the following ranges H: 0-179, S: 0-255, V: 0-255 

# red
# red
hueUp = 180
hueLow = 0
hue_range = 10

saturation = 170
saturation_range = 50

value = 100
value_range = 80
lower_red = np.array([hueUp - hue_range,saturation - saturation_range,value - value_range])
upper_red = np.array([hueUp + hue_range,saturation + saturation_range,value + value_range])

additional_lower_red = np.array([hueLow - hue_range,saturation - saturation_range,value - value_range])
additional_upper_red = np.array([hueLow + hue_range,saturation + saturation_range,value + value_range])

# green
hue = 60
hue_range = 20
saturation = 155
saturation_range = 100
value = 130
value_range = 125
lower_green = np.array([hue - hue_range,saturation - saturation_range,value - value_range])
upper_green = np.array([hue + hue_range,saturation + saturation_range,value + value_range])
# blue
# blue
hue = 100
hue_range = 30
saturation = 200
saturation_range = 70
value = 220
value_range = 80
lower_blue = np.array([hue - hue_range,saturation - saturation_range,value - value_range])
upper_blue = np.array([hue + hue_range,saturation + saturation_range,value + value_range])
# yellow
hue = 30
hue_range = 10
lower_yellow = np.array([hue - hue_range,saturation - saturation_range,value - value_range])
upper_yellow = np.array([hue + hue_range,saturation + saturation_range,value + value_range])
# white
hue = 89.5
hue_range = 89.5
saturation = 14
saturation_range = 14
value = 235
value_range = 20
lower_white = np.array([hue - hue_range,saturation - saturation_range,value - value_range])
upper_white = np.array([hue + hue_range,saturation + saturation_range,value + value_range])
# pink
hue = 10
hue_range = 10
saturation = 105
saturation_range = 50
value = 155
value_range = 100
lower_pink = np.array([hue - hue_range,saturation - saturation_range,value - value_range])
upper_pink = np.array([hue + hue_range,saturation + saturation_range,value + value_range])

# color ranges inside an array for for loop
color_lower_list = (lower_red, lower_green, lower_blue, lower_yellow, lower_white, lower_pink)
color_upper_list = (upper_red, upper_green, upper_blue, upper_yellow, upper_white, upper_pink)
### morphological operations
# optional mapping of values with morphological shapes
def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE

# dilation with parameters
def dilatation(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.dilate(img, element)

# erosion with parameters
def erosion(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.erode(img, element)

# opening
def opening(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, element)

# closing
def closing(img,size,shape): 
    element = cv2.getStructuringElement(shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, element)

# set color under test
num_colors = 6
color_names = ['red', 'green', 'blue', 'yellow', 'white','pink']


# setting the parameters that work for all colors

# set individual (per color) parameters

num_test_images_succeeded = 0
for img_name in glob.glob('images/chewing_gum_balls*.jpg'): 
    # load image
    print ('Searching for colored balls in image:',img_name)

    all_colors_correct = True

    for c in range(0,num_colors):
        
        img = cv2.imread(img_name,cv2.IMREAD_COLOR)
        height = img.shape[0]
        width = img.shape[1]

        # TODO: Insert your algorithm here
        # first convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # create a mask from current color
        mask = cv2.inRange(hsv, color_lower_list[c], color_upper_list[c])
        #red is around both 360° and 0° (179 in cv2) so 2 masks need to be combined
        if (color_names[c] == 'red'):
            mask2 = cv2.inRange(hsv, additional_lower_red, additional_upper_red)
            mask = mask | mask2

        #morphological operations
        kernel_size = 3
        kernel_shape = morph_shape(2)
        mask = opening(mask,kernel_size, kernel_shape)
        mask = closing(mask,kernel_size, kernel_shape)

        # find connected components
        connectivity = 4
        (num_Labels, labels, stats, centroids) = cv2.connectedComponentsWithStats(mask,connectivity,cv2.CV_32S) # num_labels = 0 # TODO: implement something to set this variable

        # find center of mass and draw a mark in the original image
        red_BGR = (0,0,255)
        green_BGR = (0,255,0)
        circle_size = 25
        circle_thickness = 5
        min_size = 20
        num_rejected = 1

        # go through all (reasonable) found connected components
        for i in range(1,num_Labels):
            # check size and roundness as plausibility
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            if w < min_size or h < min_size:
                print ('Found a too small component.')
                num_rejected += 1
                continue # found component is too small to be correct 
            if w > h:
                roundness = 1.0 / (w/h)
            elif h > w:
                roundness = 1.0 / (h/w)  
            if (roundness < .75):
                print ('Found a component that is not round enough.')
                num_rejected += 1
                continue # ratio of width and height is not suitable

            # find and draw center
            center = centroids[i]
            center = np.round(center)
            center = center.astype(int)
            cv2.circle(img,center,circle_size,red_BGR,circle_thickness)

            # find and draw bounding box
            cv2.rectangle(img, (x, y), (x + w, y + h), green_BGR, 3) 

        num_final_labels = num_Labels-num_rejected
        success = (num_final_labels == int(gt_list[c]))
        
        if success:
            print('We have found all', str(num_final_labels),'/',str(gt_list[c]), color_names[c],'chewing gum balls. Yeah!')
            foo = 0
        elif (num_final_labels > int(gt_list[c])):
            print('We have found too many (', str(num_final_labels),'/',str(gt_list[c]),') candidates for', color_names[c],'chewing gum balls. Damn!')
            all_colors_correct = False
        else:
            print('We have not found enough (', str(num_final_labels),'/',str(gt_list[c]),') candidates for', color_names[c],'chewing gum balls. Damn!')
            all_colors_correct = False
        
        # debug output of the test images
        if ((img_name == 'images\chewing_gum_balls01.jpg') 
            or (img_name == 'images\chewing_gum_balls04.jpg') 
            or (img_name == 'images\chewing_gum_balls06.jpg')):
            # show the original image with drawings in one window
            cv2.imshow('Original image', img)
            
            # show other images?
            cv2.imshow('Mask image',mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    if all_colors_correct:
        num_test_images_succeeded += 1
        print ('Yeah, all colored objects have been found correctly in ',img_name)

print ('Test result:', str(num_test_images_succeeded),'test images succeeded.')

        
