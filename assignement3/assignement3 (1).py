'''
Assignement 03: Hybrid
Group: <7>
Names: <Martens, Jonathan; Leisinger, Oliver>
Date: <12.11.2021>
Sources: 'https://pythonexamples.org/python-opencv-image-filter-convolution-cv2-filter2d/'
'''

import cv2
import numpy as np
from numpy.lib.type_check import imag

# global helper variables
window_width = 500
window_height = 500

# highpass-kernel from 'https://pythonexamples.org/python-opencv-image-filter-convolution-cv2-filter2d/'
hpf_kernel = np.array([
    [0.0, -1.0, 0.0],
    [-1.0, 4.0, -1.0],
    [0.0, -1.0, 0.0]
])

# Size for lowpass-kernel
lpf_ksize = 9


def LPFiltering(image):
    
    lpf_result = cv2.GaussianBlur(image, (lpf_ksize, lpf_ksize), 5)

    return lpf_result

def HPFiltering(image):

    hpf_result = cv2.filter2D(image, -1, hpf_kernel)

    return hpf_result

def getFrequencies(image):
    """ Compute spectral image with a DFT
    """
    # convert image to floats and do dft saving as complex output
    dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

    # apply shift of origin from upper left corner to center of image
    dft_shift = np.fft.fftshift(dft)

    # extract magnitude and phase images
    mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])

    # get spectrum for viewing only
    spec = ((1/20) * np.log(mag))

    # Return the resulting image (as well as the magnitude and
    # phase for the inverse)
    return spec, mag, phase

def createFromSpectrum(mag, phase):
    # convert magnitude and phase into cartesian real and imaginary components
    real, imag = cv2.polarToCart(mag, phase)

    # combine cartesian components into one complex image
    back = cv2.merge([real, imag])

    # shift origin from center to upper left corner
    back_ishift = np.fft.ifftshift(back)

    # do idft saving as complex output
    img_back = cv2.idft(back_ishift)

    # combine complex components into original image again
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # re-normalize to 8-bits
    min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
    print(min, max)
    img_back = cv2.normalize(img_back, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return img_back

def combine(lpf, hpf):
    lpf_result, lpf_mag, lpf_phase = getFrequencies(lpf)
    hpf_result, hpf_mag, hpf_phase = getFrequencies(hpf)
    mag = np.multiply(lpf_mag, hpf_mag)
    phase = np.multiply(lpf_phase, hpf_phase)
    return createFromSpectrum(mag, phase)



def main():
    """ Load an image, compute frequency domain image from it and display
    both or vice versa """
    image_name_lpf = 'assignement3\images\pic1.jpg'
    image_name_hpf = 'assignement3\images\pic2.jpg'

    # Load & resize image for lp
    image_lpf = cv2.imread(image_name_lpf, cv2.IMREAD_GRAYSCALE)
    image_lpf = cv2.resize(image_lpf, (window_width, window_height))

    # Load & resize image for hp
    image_hpf = cv2.imread(image_name_hpf, cv2.IMREAD_GRAYSCALE)
    image_hpf = cv2.resize(image_hpf, (window_width, window_height))

    # show the original image that will be LPFed
    lpf_title_original = 'Original image for LP'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(lpf_title_original, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lpf_title_original, window_width, window_height)
    cv2.imshow(lpf_title_original, image_lpf)


    # set lp-filter
    result_lpf = LPFiltering(image_lpf)

    # show the resulting image with lp
    lpf_title_result = 'Low Pass'
    
    cv2.namedWindow(lpf_title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lpf_title_result, window_width, window_height)
    cv2.imshow(lpf_title_result, result_lpf)


    # show the original image that will be HPFed
    hpf_title_original = 'Original image for HP'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(hpf_title_original, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(hpf_title_original, window_width, window_height)
    cv2.imshow(hpf_title_original, image_hpf)


    freq_hpf_add = HPFiltering(image_hpf)

    # show the resulting image with hp
    lpf_title_result = 'High Pass'

    cv2.namedWindow(lpf_title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lpf_title_result, window_width, window_height)
    cv2.imshow(lpf_title_result, freq_hpf_add)


    together = combine(result_lpf, freq_hpf_add)

    #show the hybrid image
    lpf_title_result = 'Hybrid Image'

    cv2.namedWindow(lpf_title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(lpf_title_result, window_width, window_height)
    cv2.imshow(lpf_title_result, together)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


if (__name__ == '__main__'):
    main()