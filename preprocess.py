import pydicom
import glob, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import re
from joblib import Parallel, delayed

# path to dataset in HPC
rd = '/home/santosh_lab/shared/SajeshA/rsna-2024-lumbar-spine-degenerative-classification'
N_PROCESSES = 32

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# analyzing training labels
dfc = pd.read_csv(f'{rd}/train_label_coordinates.csv')
dfc.head()
df = pd.read_csv(f'{rd}/train_series_descriptions.csv')
df.head()

# analyzing series_description
df['series_description'].value_counts()

# analyzing arbitrary study id
print(df[df['study_id']==4096820034])
print(dfc[dfc['study_id']==4096820034])

def dicom_2_png(src_path, dst_path):
    """
    Converts a DICOM image to PNG and saves it to the specified path.

    The function normalizes the pixel values to [0, 255], resizes the image to 512x512, 
    and saves it as a PNG.

    Params:
    - src_path (str): Path to the DICOM file.
    - dst_path (str): Path to save the PNG.

    Returns:
    - None

    Raises:
    - AssertionError: If the image isn't 512x512 after resizing.
"""

    dicom_data = pydicom.dcmread(src_path)
    image = dicom_data.pixel_array
    image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
    img = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)
    assert img.shape == (512, 512)
    cv2.imwrite(dst_path, img)

st_ids = df['study_id'].unique()
print(st_ids[:3])
print(len(st_ids))

desc = list(df['series_description'].unique())
print(desc)

def process_study(si):
    pdf = df[df['study_id']==si]
    for ds in desc:
        ds_ = ds.replace('/', '_')
        pdf_ = pdf[pdf['series_description']==ds]
        os.makedirs(f'{rd}/cvt_png/{si}/{ds_}', exist_ok=True)
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob.glob(f'{rd}/train_images/{row["study_id"]}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)
            
        if len(allimgs)==0:
            print(si, ds, 'has no images')
            continue

        if ds == 'Axial T2':
            for j, impath in enumerate(allimgs):
                dst = f'{rd}/cvt_png/{si}/{ds}/{j:03d}.png'
                dicom_2_png(impath, dst)
                
        elif ds == 'Sagittal T2/STIR':
            
            step = len(allimgs) / 10.0
            st = len(allimgs)/2.0 - 4.0*step
            end = len(allimgs)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'{rd}/cvt_png/{si}/{ds_}/{j:03d}.png'
                ind2 = max(0, int((i-0.5001).round()))
                dicom_2_png(allimgs[ind2], dst)
                
            assert len(glob.glob(f'{rd}/cvt_png/{si}/{ds_}/*.png'))==10
                
        elif ds == 'Sagittal T1':
            step = len(allimgs) / 10.0
            st = len(allimgs)/2.0 - 4.0*step
            end = len(allimgs)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                dst = f'{rd}/cvt_png/{si}/{ds}/{j:03d}.png'
                ind2 = max(0, int((i-0.5001).round()))
                dicom_2_png(allimgs[ind2], dst)
                
            assert len(glob.glob(f'{rd}/cvt_png/{si}/{ds}/*.png'))==10

Parallel(n_jobs=N_PROCESSES)(delayed(process_study)(si) for si in st_ids)

