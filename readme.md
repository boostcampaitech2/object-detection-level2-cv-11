# <div align='center'>Architecture<div>

`./mmdetection` : [mmdetection framework](https://github.com/open-mmlab/mmdetection)

`./yolov5` : [YOLOv5 framework](https://github.com/ultralytics/yolov5)

`./data_sub` : self-made utils

# <div align='center'>Data Preperation<div>

Aistage에서 제공하는 쓰레기 데이터를 사용합니다.
```bash
$ wget https://aistages-prod-server-public.s3.amazonaws.com/app/Competitions/000076/data/data.tar.gz
```

# <div align='center'>Quick Start<div>

### Training 
<details open>
<summary> mmdetection </summary>

- 경로 : `opt/ml/detection/mmdetection`
1. 원하는 모델과 파라마티, 하이퍼 파라미터 config 세팅
2. `work_dir`을 지정, wandb project name, entity 설정
3. `config_dir` 지정후 코드 실행

```bash
$ python tools/train.py [config_dir]
```

</details>

<details open>
<summary> YOLOv5 </summary>

- 경로 : `opt/ml/detection/yolov5`

YOLOv5 format에 맞는 데이터 필요 

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

</details>

### Inference

<details open>
<summary> mmdetection </summary>

- 경로: `opt/ml/detection/mmdetection`
1. `--config_dir` Inference할 config 선택
2. `--epoch` Inference할 저장되어있는 ,pth파일 선택

```bash
$ python inference.py --config_dir[config_dir] --epoch [pth file name]
```

</details>

<details open>
<summary> YOLOv5 </summary>

- 경로 `opt/ml/detection/yolov5`

```bash
$ python inference.py --label_dir {label_path} 
```

</details>

### pseudo
> 
- `--pre` : 기존 json파일
- `--test` : 합칠 json파일
- `--csv` : pseudo label
- `--output`: output 파일 이름
- `t--h` : confidence score

```bash
$ python makej.py --csv [csv_file] --output [output_file_name]
```

### ensemble
> 
- 경로 : `opt/ml/detection`
- `--model_dir` : 모델이 저장된 경로
- `--weights` : 각 모델의 가중치
- `--save_dir` : 앙상블 된 모델이 저장될 경로
- `--method` : 앙상블 할 방식 설정

```bash
$ python /data_sub/ensemble.py --model_dir [model_dir] --weights [weights:list] --save_dir [save_dir] --method wbf
```
  
  
# <div align='center'>Reference<div>
- https://www.kaggle.com/jakubwasikowski/stratified-group-k-fold-cross-validation
- https://www.kaggle.com/iru538/stratified-group-k-fold
