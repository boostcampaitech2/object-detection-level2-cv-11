import json
from collections import OrderedDict
import numpy as np
import pandas as pd
from collections import Counter , defaultdict
import random
from copy import deepcopy
from objdict import ObjDict
SEED=1234

def make_valid(num_split, fromdata="/opt/ml/detection/dataset/refined_train.json",
                tofolder='/opt/ml/detection/dataset/refined_kfold'):

    file_data = OrderedDict()
    #file_data['info']=OrderedDict()
    #file_data['licenses']=[]
    file_data['images']=[]
    file_data['categories']=[]
    file_data['annotations']=[]

    val_data = OrderedDict()

    with open(fromdata,'r') as f:
        json_data = json.load(f)


    #file_data['info'] = json_data['info']
    #file_data['licenses'] = json_data['licenses']
    file_data['categories'] = json_data['categories']

    val_data=deepcopy(file_data)

    df=pd.DataFrame()

    for j,i in enumerate(json_data['annotations']):
        new=[(j,i['image_id'],i['category_id'])]
        dfnew=pd.DataFrame(new,columns=['id','image_id','category_id'])
        df=df.append(dfnew, ignore_index=True)

    def stratified_group_k_fold(X, y, groups, k, seed=None):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
        
        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(k):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

    groups = df.image_id.values
    labels = df.category_id.values
    splits = list(stratified_group_k_fold(df,labels,groups,k=num_split,seed=SEED))

    for i, (train_idx, valid_idx) in enumerate(splits):
        file_data['images']=[]
        file_data['annotations']=[]

        val_data['images']=[]
        val_data['annotations']=[]

        train = df.iloc[train_idx, :]  
        valid = df.iloc[valid_idx, :]
        
        train_ids = train['image_id'].drop_duplicates().values
        valid_ids = valid['image_id'].drop_duplicates().values  
        print('train_idx', train_idx)
        print('------------------split-------------------')
        for t in train_ids:
            t=int(t)
            file_data['images'].append(json_data['images'][t])
        for t in train['id']:
            file_data['annotations'].append(json_data['annotations'][t])


        with open(tofolder+'train_fold'+str(i+1)+'.json','w',encoding="utf-8") as make_file:
            json.dump(file_data,make_file,ensure_ascii=False, indent='\t')

        for v in valid_ids:
            v=int(v)
            val_data['images'].append(json_data['images'][v])

        for v in valid['id']:
            val_data['annotations'].append(json_data['annotations'][v])


        with open(tofolder+'val_fold'+str(i+1)+'.json','w',encoding="utf-8") as val_file:
            json.dump(val_data,val_file,ensure_ascii=False, indent='\t')

        print( '=========================' )
        print( 'K=', i+1 )
        
        print( '[train]' )
        print( 'class: ', Counter(train['category_id']) )   
        print( 'images: ', len(train_ids) )   
        
        print( '===' )
        
        print( '[valid]' )
        print( 'class: ', Counter(valid['category_id']) )
        print( 'images: ', len(valid_ids) ) 

if __name__=='__main__':
    make_valid(5)
