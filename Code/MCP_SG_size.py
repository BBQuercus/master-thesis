import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.ndimage as ndi
import pandas as pd
import skimage.feature
import skimage.io
import skimage.measure


def read_files(files):
    '''
    Reads groups of image files and loads them as numpy arrays.
    '''
    files = sorted(files)
    file_order = [3, 0, 1, 2]
    min_slice = 1  # First slice ignored

    files = [files[i] for i in file_order]
    files = files[:3]
    image = list(map(skimage.io.imread, files))
    image = np.stack(image)
    image = np.max(image[:, min_slice:], axis=1)

    assert image.ndim == 3

    return image


def display_files(files):
    ''' Displays files for easier viewing. '''
    image = read_files(files)

    fig, ax = plt.subplots(1, image.shape[0], figsize=(16, 10))
    for i in range(image.shape[0]):
        ax[i].imshow(image[i])
        ax[i].set_axis_off()

    plt.show()


def nuclear_detection(image):
    ''' Detects and segments nuclear instances. '''
    img = ndi.gaussian_filter(image, 6)
    img = img > skimage.filters.threshold_otsu(img)
    otsu = img
    img = ndi.binary_erosion(img)
    img = ndi.distance_transform_edt(img)
    img = img > 5
    lab = ndi.label(img)[0]
    img = skimage.morphology.watershed(image, lab, mask=otsu)
    return img


def cytoplasmic_segmentation(image, nucleus):
    ''' Basic foreground / background segmentation. '''
    img = ndi.gaussian_filter(image, 5)
    img = img > img.mean() * 0.75
    img = skimage.morphology.watershed(img, nucleus, mask=img)
    img -= nucleus
    return img


def get_value(r, c, image, threshold=None):
    '''
    Returns the label value at a blob position.
    Optional thresholding allows for boolean predictions.
    '''

    if threshold is not None:
        image = image > threshold

    label = image[int(r), int(c)]

    return label


def main():

    ROOT = 'PATH'
    ROOT_SEG = 'PATH'  # Output from Fluffy

    files_nd = glob.glob(f'{ROOT}/*Ars*.nd')
    basenames = sorted([os.path.splitext(f)[0] for f in files_nd])

    files = []
    for basename in basenames:
        files.append(glob.glob(f'{basename}*.stk'))

    files_seg = sorted(glob.glob(f'{ROOT_SEG}/*Ars*.tiff'))
    images_seg = list(map(skimage.io.imread, files_seg))

    rows = []

    for i, file in enumerate(files):
        image = read_files(file)

        nucleus = nuclear_detection(image[0])
        cytoplasm = cytoplasmic_segmentation(image[1], nucleus)
        granules = (images_seg[i] > 0).astype(np.uint8)
        granules = np.where(nucleus > 0, 0, granules)
        granules = skimage.measure.label(granules)

        for n_granule in np.unique(granules)[1:]:
            granule = (granules == n_granule).astype(np.uint8)
            r_granule = skimage.measure.regionprops(granule, image[2])[0]
            r, c = r_granule.centroid

            row = {
                # General information
                'file': file[0],

                # Cellular measures
                'cell': get_value(r, c, cytoplasm),
                'area': r_granule.area,
                'intensity': r_granule.mean_intensity,
            }
            rows.append(pd.Series(row))

    df = pd.DataFrame(rows)
    df.to_csv('data.csv')


if __name__ == "__main__":
    main()
