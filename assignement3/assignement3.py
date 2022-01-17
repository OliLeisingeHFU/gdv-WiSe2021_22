'''
Assignement 03: <Hybrid Images>
Group: <7>
Names: <Martens, Jonathan; Leisinger, Oliver>
Date: <12.11.2021>
Sources: <https://stackoverflow.com/a/66936362, https://thispersondoesnotexist.com, /gdv tutorial 10>
'''

import cv2
import numpy as np

# global variables
window_width = 300
window_height = 300
blur_size = 21

lpf_mask_radius = 30
hpf_mask_radius = 15



def spectrumForViewing(dft_shift):
    mag = np.abs(dft_shift)
    return ((1/20) * np.log(mag))

def LPFiltering(image):
    # do dft saving as complex output
    dft = np.fft.fft2(image, axes=(0,1))

    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)

    # create circle mask
    mask = np.zeros_like(image)
    cx = mask.shape[1] // 2
    cy = mask.shape[0] // 2
    cv2.circle(mask, (cx,cy), lpf_mask_radius, (255,255,255), -1)[0]

    # blur the mask
    mask = cv2.GaussianBlur(mask, (blur_size,blur_size), blur_size)

    # apply mask to dft_shift
    dft_shift_masked = np.multiply(dft_shift,mask) / 255

    # show the resulting image
    title_result = 'Low Frequencies image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_result, window_width, window_height)
    cv2.imshow(title_result, (spectrumForViewing(dft_shift_masked)))
    
    return dft_shift_masked


def HPFiltering(image):
    # do dft saving as complex output
    dft = np.fft.fft2(image, axes=(0,1))

    # apply shift of origin to center of image
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros_like(image)
    cx = mask.shape[1] // 2
    cy = mask.shape[0] // 2
    cv2.circle(mask, (cx,cy), hpf_mask_radius, (255,255,255), -1)[0]
    mask = 255 - mask

    # blur the mask
    mask = cv2.GaussianBlur(mask, (blur_size,blur_size), 0)

    # apply mask to dft_shift
    dft_shift_masked = np.multiply(dft_shift,mask) / 255

    # show the resulting image
    title_result = 'High Frequencies image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_result, window_width, window_height)
    cv2.imshow(title_result, (spectrumForViewing(dft_shift_masked)))

    return dft_shift_masked

def combine(lpf, hpf):

    lpf_dft = LPFiltering(lpf)
    hpf_dft = HPFiltering(hpf)

    combined_dft = lpf_dft + hpf_dft
    # show the resulting image
    title_result = 'Combined Frequencies image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_result, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_result, window_width, window_height)
    cv2.imshow(title_result, (spectrumForViewing(combined_dft)))

    back_ishift_combined = np.fft.ifftshift(combined_dft)
    hybrid_image = np.fft.ifft2(back_ishift_combined, axes=(0,1))
    hybrid_image = np.abs(hybrid_image).clip(0,255).astype(np.uint8)
    return hybrid_image


def main():
    image_name_lpf_1 = 'assignement3/images/pic1.jpg'
    image_name_hpf_1 = 'assignement3/images/pic2.jpg'

    image_name_lpf_2 = 'assignement3/images/pic3.jpg'
    image_name_hpf_2 = 'assignement3/images/pic4.jpg'

    # Load the image. Change image_name_lpf_1  to  image_name_lpf_2  and  image_name_hpf_1  to  image_name_hpf_2 
    # to load the second result
    image_lpf = cv2.imread(image_name_lpf_2, cv2.IMREAD_GRAYSCALE)
    image_lpf = cv2.resize(image_lpf, (window_width, window_height))
    image_hpf = cv2.imread(image_name_hpf_2, cv2.IMREAD_GRAYSCALE)
    image_hpf = cv2.resize(image_hpf, (window_width, window_height))

    # show the original image that will be LPFed
    title_original = 'Original image for LPF'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_original, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_original, window_width, window_height)
    cv2.imshow(title_original, image_lpf)

    # show the original image that will be HPFed
    title_original = 'Original image for HPF'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_original, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_original, window_width, window_height)
    cv2.imshow(title_original, image_hpf)

    # and compute image back from frequencies
    title_back = 'Hybrid image'
    # Note that window parameters have no effect on MacOS
    cv2.namedWindow(title_back, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title_back, window_width, window_height)
    cv2.imshow(title_back, combine(image_lpf, image_hpf))

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


if (__name__ == '__main__'):
    main()