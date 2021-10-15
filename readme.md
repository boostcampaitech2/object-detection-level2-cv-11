# 사용법

> Training
> 
- 경로 : `opt/ml/detection/mmdetection`
1. 원하는 모델과 파라마티, 하이퍼 파라미터 config 세팅
2. `work_dir`을 지정, wandb project name, entity 설정
3. `config_dir` 지정후 코드 실행

```python
python tools/train.py [config_dir]
```

> Inference
> 
- 경로: `opt/ml/detection/mmdetection`
1. `--config_dir` Inference할 config 선택
2. `--epoch` Inference할 저장되어있는 ,pth파일 선택

```python
python inference.py --config_dir[config_dir] --epoch [epoch.pth_dir]
```

> pseudo
> 
- `--pre` : 기존 json파일
- `--test` : 합칠 json파일
- `--csv` : pseudo label
- `--output`: output 파일 이름
- `t--h` : confidence score

```python
python makej.py --csv [csv_file] --output [output_file_name]
```

> ensemble
> 
- 경로 : `opt/ml/detection`
- `--model_dir` : 모델이 저장된 경로
- `--weights` : 각 모델의 가중치
- `--save_dir` : 앙상블 된 모델이 저장될 경로
- `--method` : 앙상블 할 방식 설정

```python
python /data_sub/ensemble.py --model_dir [model_dir] --weights [weights:list] --save_dir [save_dir] --method wbf
```