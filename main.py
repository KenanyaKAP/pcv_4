import cv2
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def ideal_filter(img, low_cut, high_cut, inverted):
    M, N = img.shape
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    u = np.arange(M)
    v = np.arange(N)
    u = u - M//2
    v = v - N//2
    U, V = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    mask = ~((D < high_cut) & (D >= low_cut)) if inverted else ((D < high_cut) & (D >= low_cut))

    dft_shift_filtered = dft_shift * mask[:, :, np.newaxis]

    idft = np.fft.ifftshift(dft_shift_filtered)
    img_filtered = cv2.idft(idft)
    img_filtered = cv2.magnitude(img_filtered[:, :, 0], img_filtered[:, :, 1])

    img_filtered = cv2.normalize(img_filtered, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return img_filtered, dft_shift, dft_shift_filtered, mask

def plot_spectrum(dft_shift):
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1] + 1e-5))
    return magnitude_spectrum

img = cv2.imread('Tower2ITS.jpg', cv2.IMREAD_GRAYSCALE)

lowCut = 0
highCut = img.shape[0] + 7
invert = 0

fig, axs = plt.subplots(2, 2, figsize=(12, 6))

def updateImage():
    img_filtered, dft_before, dft_after, mask = ideal_filter(img, lowCut, highCut, invert)
    
    # Plot original image
    axs[0, 0].cla(), axs[0, 0].set_title('Original')
    axs[0, 0].imshow(img, cmap='gray'), axs[0, 0].axis('off')
    
    # Plot frequency spectrum
    axs[0, 1].cla(), axs[0, 1].set_title('Frequency Spectrum')
    axs[0, 1].imshow(plot_spectrum(dft_before), cmap='gray'), axs[0, 1].axis('off')
    
    # Plot Mask
    axs[1, 0].cla(), axs[1, 0].set_title('Mask')
    axs[1, 0].imshow(mask, cmap='gray'), axs[1, 0].axis('off')
    
    # Plot frequency spectrum after masked
    axs[1, 1].cla(), axs[1, 1].set_title('Frequency Spectrum after masked')
    axs[1, 1].imshow(plot_spectrum(dft_after), cmap='gray'), axs[1, 1].axis('off')

    # Convert histogram canvas to image and show
    fig.canvas.draw()
    histoImg = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    histoImg = histoImg.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    histoImg = cv2.cvtColor(histoImg,cv2.COLOR_RGB2BGR)
    cv2.imshow("Histogram",histoImg)
    
    # Show image
    cv2.imshow('Filtered', img_filtered)

updateImage()

def changeLowCutFreq(x):
    global lowCut
    lowCut = x
    cv2.setTrackbarMin('High Cut Freq', 'Filtered', lowCut + 1)
    updateImage()

def changeHighCutFreq(x):
    global highCut
    highCut = x
    cv2.setTrackbarMax('Low Cut Freq', 'Filtered', min(highCut + 1, img.shape[0] + 6))
    updateImage()

def changeInverted(x):
    global invert
    invert = x
    updateImage()

cv2.createTrackbar('Low Cut Freq', 'Filtered', 0, highCut, lambda x: changeLowCutFreq(x))
cv2.createTrackbar('High Cut Freq', 'Filtered', highCut, highCut, lambda x: changeHighCutFreq(x))
cv2.createTrackbar('Inverted', 'Filtered', 0, 1, lambda x: changeInverted(x))

while 1:
    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()