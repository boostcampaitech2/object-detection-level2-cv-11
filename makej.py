import argparse
import os
import numpy as np
import json
import pandas as pd


def main(args):

    jsonFile =f'/opt/ml/detection/dataset/{args.pre}.json'
    with open(jsonFile) as file:
        pre = json.load(file)

    testFile = f'/opt/ml/detection/dataset/{args.test}.json'
    with open(testFile) as f:
        t = json.load(f)

    imid=4883 #train images

    for a,b in enumerate(t['images']):
        b['id']=imid+a
        pre['images'].append(b)

    tid=23144 #train annotations
    test=pd.read_csv(f'{args.csv}')


    for t,b in zip(test.iterrows(),t['images']):
        i,v=t    

        pred=list(str(v[0]).split(" "))

        if '' in pred:
            pred.remove('')
        predict=[pred[i:i+6] for i in range(0,len(pred),6)]
        score=[pred[i+1] for i in range(0,len(pred),6)]

        score=map(float,score)
        ms=max(score)
        if ms<args.thr:
            score=ms
        else:
            score=args.thr

        for j in predict:
            if len(j)==6 and float(j[1])>=score:
                dic={}
                dic['image_id']=imid
                dic['category_id']=int(j[0])
                dic['area']=round(float(j[4])*float(j[5]),2)
                dic['bbox']=[round(float(j[2]),2),round(float(j[3]),2),round(float(j[4])-float(j[2]),2),round(float(j[5])-float(j[3]),2)]
                dic['iscrowd']=0
                dic['id']=tid
                tid+=1
                pre['annotations'].append(dic)
            # else:
        imid+=1
            
    with open(f'/opt/ml/detection/dataset/{args.output}.json','w',encoding="utf-8") as mf:
        json.dump(pre,mf,ensure_ascii=False,indent="\t")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--pre",
        type=str,
        default='train'
    )
    parser.add_argument(
        "--test",
        type=str,
        default='test'
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None
    )
    parser.add_argument(
        "--thr",
        type=float,
        default=0.3
    )

    args = parser.parse_args()

    if args.csv is None:
        raise NameError('set csv')
    if args.output is None:
        raise NameError('set output json file name')
    
    main(args)
