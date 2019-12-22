from inferens import RemoveBackground
import argparse
import pandas as pd
import os
import cv2
import numpy as np
def Car_Segmenatation(path_in, path_out, path_model,threshold,image_size):
    mask_predictor = RemoveBackground(path_model,image_size)

    data = pd.DataFrame({"image":os.listdir(path_in)})
    for image_file in data.iloc[:,0]:
        image = cv2.resize(cv2.imread(os.path.join(path_in,image_file)), (image_size,image_size))
       	probability = mask_predictor.predict(image, threshold=0.5)
        image[probability==0]=[255,255,255]
        cv2.imwrite(os.path.join(path_out,image_file),image)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'folders to files')
    parser.add_argument('path_in', type=str, help='dir with image')
    parser.add_argument('path_out', type=str, help='output dir')
    parser.add_argument('path_model', type = str, help = 'path_model')
    parser.add_argument('threshold', type = float, help = 'threshold')

    parser.add_argument('image_size', type = int, help = 'image_size')
    args = parser.parse_args()
    Car_Segmenatation(args.path_in,args.path_out,args.path_model,args.threshold,args.image_size)
