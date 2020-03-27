from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import pandas as pd
import scipy.ndimage as ndi
import skimage.feature
import skimage.io
import skimage.measure


LOG_FORMAT = "%(levelname)s %(asctime)s - %(filename)s %(funcName)s %(lineno)s - %(message)s"
logging.basicConfig(filename="./treat.log",
                    level=logging.DEBUG,
                    format=LOG_FORMAT,
                    filemode="a")
log = logging.getLogger()


def read_files(files):
    ''' Reads groups of image files and loads them as numpy arrays. '''
    files = sorted(files)
    file_order = [2, 1, 0]
    min_slice = 1  # First slice ignored

    files = [files[i] for i in file_order]
    image = list(map(skimage.io.imread, files))
    image = np.stack(image)
    image = np.max(image[:, min_slice:], axis=1)

    assert image.ndim == 3

    return image


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


def nuclear_filtering(image, label):
    ''' Filters out wrongly segmented nuclei from a labeled image. '''

    import seaborn as sns
    reg = skimage.measure.regionprops(label, image)
    hist_intensity = sorted([r.mean_intensity for r in reg])
    hist_area = sorted([r.area for r in reg])

    sns.lineplot(hist_intensity, np.arange(len(hist_intensity)))
    sns.lineplot(hist_area, np.arange(len(hist_area)))

    plt.axvline(hist_intensity[2], color='red')
    plt.show()


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
    blobs = skimage.feature.blob_log(
        image,
        max_sigma=2,
        threshold=0.05)
    return blobs


def get_closest_blob(x1, y1, x2, y2, cutoff=5):
    '''
    Returns a list of the distance to the closest coordinates.
    Closes coordinates are measured between all coordinates in x1/y1
    relative to x2/y2. Only coordinates within a distance cutoff will be
    measured. If no coordinates within this cutoff is found,
    the distance is set to None.

    Lowering this cutoff decreases computational complexity but
    if set too small might not detect nearby blobs.
    '''

    dists = []

    for cx1, cy1 in zip(x1, y1):
        area = ((x2 < cx1 + cutoff) & (x2 > cx1 - cutoff) |
                (y2 < cy1 + cutoff) & (y2 > cy1 - cutoff))

        if sum(area) == 0:
            dists.append(-1)
            continue

        ax2 = x2[area]
        ay2 = y2[area]

        cdists = [
            np.linalg.norm([cx2 - cx1, cy2 - cy1])
            for cx2, cy2 in zip(ax2, ay2)
        ]
        dists.append(min(cdists))

    return dists


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


def blob_visualization(fname, image, blobs, size=False):
    ''' Shows blob detected spots on an image. '''

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Matplotlib functions plot in xy direction, not rc
    if size:
        import math
        blobs[:, 2] = blobs[:, 2] * math.sqrt(2)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color='yellow', linewidth=1, fill=False)
            ax.add_patch(c)
    else:
        ax.scatter(blobs[:, 1], blobs[:, 0], s=1, marker='x', c='yellow')

    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def main():
    ROOT = 'PATH'

    files_nd = glob.glob(f'{ROOT}/*.nd')
    basenames = [os.path.splitext(f)[0] for f in files_nd]

    files = []
    for basename in basenames:
        files.append(glob.glob(f'{basename}_w*.stk'))

    rows_blobs = []
    rows_cells = []
    for bname, file in tqdm(zip(basenames, files), desc='Files'):
        # Import
        log.info(f"Reading files: {file}.")
        image = read_files(file)

        # Segmentation
        nucleus = nuclear_detection(image[0])
        skimage.io.imsave(f"{bname}_seg_nucleus.png", nucleus, check_contrast=False)
        cytoplasm = cytoplasmic_segmentation(image[1], nucleus)
        skimage.io.imsave(f"{bname}_seg_cytoplasm.png", cytoplasm, check_contrast=False)
        cell = nucleus + cytoplasm
        log.info(f"Cells segmented and saved.")

        # Blob detection - Note rc=yx
        blobs_c1 = blob_detection(image[1])
        blob_visualization(f"{bname}_blobs_c1.png", image[1], blobs_c1)
        y1 = blobs_c1[:, 0]
        x1 = blobs_c1[:, 1]
        log.info(f"Blobs C1 detected.")

        blobs_c2 = blob_detection(image[2])
        blob_visualization(f"{bname}_blobs_c2.png", image[2], blobs_c2)
        y2 = blobs_c2[:, 0]
        x2 = blobs_c2[:, 1]
        log.info(f"Blobs C2 detected.")

        # Blob linking - Note c2=MS2, c1=Renilla
        dist_21 = get_closest_blob(x2, y2, x1, y1)
        dist_12 = get_closest_blob(x1, y1, x2, y2)
        log.info(f"Blobs linked.")

        # Blob Output - 21
        for n, d in tqdm(enumerate(dist_21), desc='Blobs', leave=False):
            cx2 = x2[n]
            cy2 = y2[n]

            row_blob = {
                # General information
                'file': file[0],
                'direction': 21,

                # Cellular measures
                'cell': get_value(cx2, cy2, cell),
                'nuclear': get_value(cx2, cy2, nucleus, threshold=0),

                # Blob measures
                'coord_x': cx2,
                'coord_y': cy2,
                'dist': d,
            }
            rows_blobs.append(pd.Series(row_blob))
        log.info(f"Blob rows computed 21.")

        # Blob Output - 12
        for n, d in tqdm(enumerate(dist_12), desc='Blobs', leave=False):
            cx1 = x1[n]
            cy1 = y1[n]

            row_blob = {
                # General information
                'file': file[0],
                'direction': 12,

                # Cellular measures
                'cell': get_value(cx1, cy1, cell),
                'nuclear': get_value(cx1, cy1, nucleus, threshold=0),

                # Blob measures
                'coord_x': cx1,
                'coord_y': cy1,
                'dist': d,
            }
            rows_blobs.append(pd.Series(row_blob))
        log.info(f"Blob rows computed.")


        # Cellular output
        for c in tqdm(np.unique(cell), desc='Cells', leave=False):
            reg = (cell == c).astype(int)
            reg_nucleus = (nucleus == c).astype(int)
            prop = skimage.measure.regionprops(reg)[0]
            prop_nucleus = skimage.measure.regionprops(reg_nucleus)[0]

            row_cell = {
                # General information
                'file': file[0],

                # Cellular measures
                'cell': c,
                'nucleus_area': prop.area,
                'cytoplasm_area': prop_nucleus.area,

                # Blob measures - Note inverse due to subreg subtraction
                'blobs_nucleus_c1': get_count(x1, y1, reg, cytoplasm),
                'blobs_cytoplasm_c1': get_count(x1, y1, reg, nucleus),
                'blobs_nucleus_c2': get_count(x2, y2, reg, cytoplasm),
                'blobs_cytoplasm_c2': get_count(x2, y2, reg, nucleus),
            }
            rows_cells.append(pd.Series(row_cell))
        log.info(f"Cellular rows computed.")

    df_blobs = pd.DataFrame(rows_blobs)
    df_blobs.to_csv('data_blobs.csv')

    df_cells = pd.DataFrame(rows_cells)
    df_cells.to_csv('data_cells.csv')


if __name__ == "__main__":
    main()
