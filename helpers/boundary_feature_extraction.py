import numpy as np
import cv2
from skimage.measure import regionprops
from skimage.draw import polygon_perimeter
from scipy.spatial import distance

# from scipy.fftpack import fft
from scipy.interpolate import splprep, splev


def extract_boundary_features(boundary_coords):
    # Compute Hu Moments
    moments = cv2.moments(boundary_coords)
    hu_moments = cv2.HuMoments(moments).flatten()

    # # Compute Zernike Moments
    # region = regionprops(boundary_coords.astype(int))[0]
    # zernike_moments = cv2.zernike(region.image, 8).flatten()

    # # Compute Fourier Descriptors
    # contour = np.squeeze(boundary_coords)
    # contour = np.array(contour, dtype=np.float32)
    # contour = cv2.approxPolyDP(contour, 0.1, True)
    # contour = np.squeeze(contour)
    # contour_complex = np.empty(contour.shape[:-1], dtype=complex)
    # contour_complex.real, contour_complex.imag = contour[:, 0], contour[:, 1]
    # fourier_desc = np.fft.fft(contour_complex)
    # fourier_desc = np.fft.fftshift(fourier_desc)
    # fourier_desc = np.abs(fourier_desc)
    # fourier_desc = fourier_desc[:len(fourier_desc)//2]

    # # Compute Polygonal Approximation
    # perimeter = polygon_perimeter(boundary_coords[:, 0], boundary_coords[:, 1])
    # perimeter = np.array(perimeter).T
    # perimeter = np.squeeze(perimeter)
    # tck, u = splprep(perimeter.T, s=0, per=True)
    # x, y = splev(np.linspace(0, 1, 100), tck)
    # poly_approx = np.vstack((x, y)).T
    # poly_approx_dist = distance.cdist(boundary_coords, poly_approx)
    # poly_approx_dist = np.min(poly_approx_dist, axis=1)

    # # Concatenate feature vectors
    # features = np.concatenate((hu_moments, zernike_moments, fourier_desc, poly_approx_dist))

    return hu_moments
