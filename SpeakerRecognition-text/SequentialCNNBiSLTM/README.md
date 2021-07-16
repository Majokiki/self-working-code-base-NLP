### Requirements

```
torch
torchtext==0.8.1
```

### Train

```bash
python main_seq.py --train=true --save_model_dir=checkpoints --lr=0.01 --epoch=50 --batch_size=16 --gpu=1
```

### Test

```bash
python main_seq.py --predict_file_path=data/train/chunyu-dialog.eval --gpu=1 --load_model_dir=best_model
```

### Predict

```bash
python predict.py --asr_file_path data/asr_test_file.txt
```
