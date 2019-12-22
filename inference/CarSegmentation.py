from inferens import RemoveBackground
import argparse
import pandas as pd
import glob
import cv2
def Car_Segmenatation(path_in, path_out, threshold,path_model,image_size):
    mask_predictor = RemoveBackground(path_model,image_size)

    data = pd.DataFrame({"image":glob.glob(path_in)})
    for image in data.iloc[:,0]:
       	mask = mask_predictor.predict(image, threshold=0.5)
        image[probability==0]=[255,255,255]
        cv2.imwrite(os.path.join(path_out,image),image)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'folders to files')
    parser.add_argument('path_in', type=str, help='dir with image')
    parser.add_argument('path_out', type=str, help='output dir')
    parser.add_argument('path_model', type = str, help = 'path_model')
    parser.add_argument('threshold', type = float, help = 'threshold')

    parser.add_argument('image_size', type = float, help = 'image_size')
    args = parser.parse_args()
    Car_Segmenatation(args.path_in,args.path_out,args.path_model,args.threshold,args.image_size)
