To use, simply extract all files from the ZIP. Then run assignement2.py in Visual Studio Code. For this you need Python 3.9.7 as well as OpenCV 4.5 installed.
To add more pictures, put them in the images folder that was extracted from the zip. They need to start with "chewing_gum_balls" and end with ".jpg".
The quickest way to cycle through the colors is to press "q".

At first, the range of each color will be calculated. Next, it will load all pictures and for each
picture it will do the following for each color (red, green, blue, yellow, white, pink):
1. Create a mask using the current color range and the current picture converted to HSV.
    red is special, as it is found both at the beginning (from around 0°-20°) and the end (from around 340°-360) of the spectrum(cv2 halves these values)
2. then morphological operations are used, turning the blobs visible on the mask more into circles.
3. Next, all Components are counted, labeled, their stats are measured and the center point of each is determined
4. The stats of each Component is checked to determine, whether it is round and big enough to be a gumball.
5. if it is accepted, a red circle will be drawn around the center point and a green box around the overall shape.
6. Lastly the image as well as the mask will be displayed (q to close) and in the terminal a message will be printed, how many balls where found of the current color.