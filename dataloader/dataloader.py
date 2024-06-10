
import large_image
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import pandas as pd
import os
from sklearn import preprocessing

OPENSLIDE_PATH = r'C:\openslide-win64-20231011\bin'


if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

class SvsLoader:
    
    
    def __init__(self, img_path,csv_file,source_folder):
         self.img_path=img_path
         self.csv_file=csv_file
         self.source_folder=source_folder
         
         
         
    def extract_level(self,img_path):
        ts = large_image.getTileSource(img_path)
        im_low_res, _ = ts.getRegion(
        scale=dict(magnification=1.25),
        format=large_image.tilesource.TILE_FORMAT_NUMPY,
    )

        im_low_res_array = np.array(im_low_res)
        #print(im_low_res_array)
        return im_low_res_array
    
    
    
    def map_csv_data_to_images(self):
        df=pd.read_csv(self.csv_file)
        df=df.dropna()
        new_size = (256,256)
        count=0
        limages=[]
        labels=[]
        for index, row in df.iterrows():
            image_filename = row['pros_heslide_img_name']
            label = row['pros_heslide_img_worst_class']
            plco_id_i=row['plco_id']
            source_path = os.path.join(self.source_folder, image_filename)
            if os.path.exists(source_path):
                limage=[]
                count+=1
                #print(count)
                #print(source_path)
                limage=self.extract_level(source_path)
                resized_image = resize(limage, new_size, anti_aliasing=True)
                limages=np.append(limages,resized_image)
                #print(label)
                labels=np.append(labels,label)
                fin_array=np.array(limages)
                labels=np.array(labels)
                fin_array=fin_array.reshape(count,256,256,3)
                le = preprocessing.LabelEncoder()
                labels=le.fit_transform(labels)
        return fin_array,labels
    