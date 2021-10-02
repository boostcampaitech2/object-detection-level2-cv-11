import pandas as pd
import os
from ensemble_boxes import *
import numpy as np
import argparse

def main(args):
    model_path = args.model_dir
    ensemble_list = os.listdir(model_path)
    models = [pd.read_csv(os.path.join(model_path, _)) for _ in ensemble_list]

    if args.weights != None and len(args.weights) != len(models):
        raise NameError('not match number of model and weights')
    
    boxes_list = [[[] for i in range(len(models))] for _ in range(4871)]
    scores_list = [[[] for i in range(len(models))] for _ in range(4871)]
    labels_list = [[[] for i in range(len(models))] for _ in range(4871)]

    for model_num, model in enumerate(models):
        for img_num, img in enumerate(model['PredictionString']):
            try:
                img = img.split()
            except:
                continue
            for pos in range(0, len(img), 6):
                obj = list(map(float, img[pos:pos+6]))
                label, score, box = int(obj[0]), float(obj[1]), np.array(obj[2:]) / 1024
                
                boxes_list[img_num][model_num].append(box)
                scores_list[img_num][model_num].append(score)
                labels_list[img_num][model_num].append(label)
    print("success make each model to list")

    for i in range(len(boxes_list)):
        boxes, scores, labels = weighted_boxes_fusion(
                                                    boxes_list[i], scores_list[i], labels_list[i], weights=args.weights,
                                                    iou_thr=args.iou_thr, skip_box_thr=args.skip_box_thr
                                                    )
        boxes_list[i], scores_list[i], labels_list[i] = boxes * 1024, scores, labels 
    print("success to ensemble each model")

    print("make csv file...")
    predictionstring = []
    file_names = []
    for img_predict in range(len(boxes_list)):
        file_names.append(models[0]['image_id'][img_predict])
        prediction_string = ''
        for label, score, box in zip(labels_list[img_predict], scores_list[img_predict], boxes_list[img_predict].tolist()):
            prediction = [str(int(label))] + [str(score)] + list(map(str, box))
            prediction_string += ' '.join(prediction) + ' '
        predictionstring.append(prediction_string)

    file_name = '_'.join([i.replace('.csv', '') for i in ensemble_list])
    submission = pd.DataFrame()
    submission['PredictionString'] = predictionstring
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(args.save_dir, f'{file_name}.csv'), index=None)
    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help='set your models dir', default='./models')
    parser.add_argument("--save_dir", type=str, help='set save dir of ensemble.csv', default='./ensemble')
    parser.add_argument("--iou_thr", type=float, help='set intersection over union for boxes to be match', default=0.4)
    parser.add_argument("--skip_box_thr", type=float, help='skip boxes with confidence', default=0.0001)
    parser.add_argument("--weights", type=eval, help='set weight of each models', default=None)

    args = parser.parse_args()
    main(args)

