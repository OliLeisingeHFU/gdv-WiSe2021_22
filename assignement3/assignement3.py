'''
Assignement 03: Hybrid
Group: <7>
Names: <Martens, Jonathan; Leisinger, Oliver>
Date: <12.11.2021>
'''

import cv2
import numpy as np

# global helper variables
window_width = 1024
window_height = 1024

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

def LPFiltering(dft_shift):
    # create circle mask
    radius = 32
    mask = np.zeros((window_width, window_height))
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]

    # blur the mask
    mask = cv2.GaussianBlur(mask, (19,19), 0)

    # apply mask to dft_shift
    dft_shift_masked = np.multiply(dft_shift,mask) / 255

    return dft_shift_masked


def HPFiltering():
    d

def combine(lpf, hpf):
    filtered_lpf = LPFiltering(lpf)
    filtered_hpf = LPFiltering(hpf)

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

def main():
    image_name_lpf = 'images\\chewing_gum_balls01.jpg'
    image_name_hpf = 'images\\chewing_gum_balls02.jpg'

    # Load the image.
    image_lpf = cv2.imread(image_name_lpf, cv2.IMREAD_GRAYSCALE)
    image_lpf = cv2.resize(image_lpf, (window_width, window_height))
    image_hpf = cv2.imread(image_name_hpf, cv2.IMREAD_GRAYSCALE)
    image_hpf = cv2.resize(image_hpf, (window_width, window_height))

    # show the original image that will be LPFed
    title_original = 'Original image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_original, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_original, window_width, window_height)
    cv2.imshow(title_original, image_lpf)

    result_lpf, mag, phase = getFrequencies(image_lpf)

    # show the resulting image
    title_result = 'Low Frequencies image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_result, window_width, window_height)
    cv2.imshow(title_result, result_lpf)

    # show the original image that will be HPFed
    title_original = 'Original image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_original, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_original, window_width, window_height)
    cv2.imshow(title_original, image_hpf)

    result_hpf, mag, phase = getFrequencies(image_hpf)

    # show the resulting image
    title_result = 'High Frequencies image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_result, window_width, window_height)
    cv2.imshow(title_result, result_hpf)

    mag, phase = combine(result_lpf, result_hpf)

    end_result = createFromSpectrum(mag, phase)
    # and compute image back from frequencies
    title_back = 'Reconstructed image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_back, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_back, window_width, window_height)
    cv2.imshow(title_back, end_result)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


if (__name__ == '__main__'):
    main()