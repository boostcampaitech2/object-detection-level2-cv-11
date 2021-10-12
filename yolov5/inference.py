import argparse
import os
import torch
import cv2
import pandas as pd
from PIL import Image
import pandas as pd

def main(args):

    dir_path = args.label_dir
    file_list = os.listdir(dir_path)
    file_list.sort()
    column_names = ['class','x_center','y_center','width','height','confidence']
    img_list = os.listdir('/opt/ml/detection/dataset/test')
    img_list.sort()
    prediction_strings = []
    file_names = []
    
    for i, name in enumerate(img_list):
        prediction_string= ''
        label = name[:-3]+'txt'
        if label not in file_list:
            print(name+"is not in file list!!!")
            prediction_strings.append(prediction_string)
            file_names.append('test/'+name)
            continue
        df = pd.read_csv(os.path.join(dir_path, label),sep=' ',names=column_names)
        for j in range(len(df)):
            xmin = (float(df['x_center'][j]) - (float(df['width'][j])/2)) * 1024
            ymin = (float(df['y_center'][j]) - (float(df['height'][j])/2)) * 1024
            xmax = (float(df['x_center'][j]) + (float(df['width'][j])/2)) * 1024
            ymax = (float(df['y_center'][j]) + (float(df['height'][j])/2)) * 1024

            prediction_string += str(df['class'][j]) + ' ' + str(
                df['confidence'][j]) + ' ' + str(round(xmin,5)) + ' ' + str(
                    round(ymin,5)) + ' ' + str(round(xmax,5)) +' ' + str(
                        round(ymax,5)) + ' '
        
        prediction_strings.append(prediction_string)
        file_names.append('test/'+ name)
        

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    if args.save_dir is not None:
        submission.to_csv(os.path.join(args.save_dir,'result.csv'),index=None)
    submission.to_csv('result.csv',index=None)
    print(submission.head())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None
    )
    args = parser.parse_args()

    if args.label_dir is None:
        raise NameError('set label directory path')
    main(args)