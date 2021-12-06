import cv2
import numpy as np

# Make blank image
img = np.zeros((1,255,1), dtype=np.uint8)

# Make gradient
for i in range (255):
    img[0][i] = i

# Resize for better visibility
new_size = (765, 200)
img = cv2.resize(img, new_size)

# copy a box from the center
box = img[0:40,362:402]

# paste the box into the dark area
img[10:50,10:50] = box

# paste the box into the bright area
img[10:50,715:755] = box

#save the generated image in the current folder
cv2.imwrite('grayscale-illusion.jpg', img)

# Show image in window like tutorial
title = 'OpenCV Python Tutorial'
cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE) # Note that window parameters have no effect on MacOS
cv2.imshow(title, img)
cv2.waitKey(0)
cv2.destroyAllWindows()