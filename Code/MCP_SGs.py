import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
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


def blob_detection(image):
    '''
    Detects spots in an image returning the coordinates and size.
    Returns in the format "row (y), column (x), sigma"
    '''
    blobs = skimage.feature.blob_log(image, max_sigma=2, threshold=0.05)
    return blobs


def blob_visualization(image, blobs, size=False):
    ''' Shows blob detected spots on an image. '''

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    # Matplotlib functions plot in xy direction, not rc
    if size:
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)

        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
            ax.add_patch(c)
    else:
        ax.scatter(blobs[:, 1], blobs[:, 0], s=1, marker='x', c='red')

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def get_value(x, y, image, threshold=None):
    '''
    Returns the label value at a blob position.
    Optional thresholding allows for boolean predictions.
    '''

    if threshold is not None:
        image = image > threshold

    label = image[int(y), int(x)]

    return label


def get_count(x, y, region, subregion=None):
    '''
    Returns the number of blobs in the specified region / subregion.
    '''

    if subregion is not None:
        region = region - subregion > 0
    else:
        region = region > 0

    x_int = x.astype(int)
    y_int = y.astype(int)
    xy_true = [region[j, i] for i, j in zip(x_int, y_int)]
    count = np.count_nonzero(xy_true)

    return count


def main():

    ROOT = 'PATH'
    ROOT_SEG = 'PATH'  # Output from Fluffy

    files_nd = glob.glob(f'{ROOT}/*Ars*.nd')
    basenames = sorted([os.path.splitext(f)[0] for f in files_nd])

    files = []
    for basename in basenames:
        files.append(glob.glob(f'{basename}*.stk'))

    files_seg = sorted(glob.glob(f'{ROOT}/*Ars*.tiff'))
    images_seg = list(map(skimage.io.imread, files_seg))

    rows_blobs = []

    for i, file in enumerate(files):
        image = read_files(file)

        nucleus = nuclear_detection(image[0])
        cytoplasm = cytoplasmic_segmentation(image[1], nucleus)
        cell = nucleus + cytoplasm
        granules = images_seg[i] > 0

        spots = blob_detection(image[2])

        for spot in spots:
            x = spot[0]
            y = spot[1]

            row_blob = {
                # General information
                'file': file[0],

                # Cellular measures
                'cell': get_value(x, y, cell),
                'nuclear': get_value(x, y, nucleus, threshold=0),
                'granular': get_value(x, y, granules),
                'granule': get_value(x, y, image[1]),

                # Blob measures
                'coord_x': spot[0],
                'coord_y': spot[1],
            }
            rows_blobs.append(pd.Series(row_blob))

    df = pd.DataFrame(rows_blobs)
    df.to_csv('data_blobs.csv')


if __name__ == "__main__":
    main()