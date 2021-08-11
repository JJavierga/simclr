# SimCLR - A Simple Framework for Contrastive Learning of Visual Representations

## Easy training

Use run.py

```
python run.py --mode=train_then_eval --train_mode=finetune   --fine_tune_after_block=4 --zero_init_logits_layer=True   --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.001   --train_epochs=1200 --train_batch_size=8 --warmup_epochs=0   --data_path=/media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/multiclassifier/Splits/30/Swin_structure --image_size=299 --eval_split=test --resnet_depth=50   --checkpoint=/home/grvc/programming/ml/simclr/tf2/checkpoints/original/simclrv2_pretrained_r50_1x_sk1_model.ckpt-250228 --model_dir=/media/grvc/MAMG_ViGUS/GRVC/Parameters/Paper/Simclr/30 --use_tpu=False --eval_split val
```


## Evaluating

Use testing.py.
data_path has two folders: train and val. Put your testing images in val (each class in a different folder).
Attention: folder /media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/multiclassifier/Splits/Test/swin/without/ has 5 classes and folder /media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/multiclassifier/Splits/Test/swin/with_general_defect has 7.

```
python testing.py --mode=eval --train_mode=finetune   --fine_tune_after_block=4 --zero_init_logits_layer=True   --global_bn=False --optimizer=momentum --learning_rate=0.1 --weight_decay=0.001   --train_epochs=1200 --train_batch_size=6 --warmup_epochs=0   --data_path=/media/grvc/MAMG_ViGUS/GRVC/Datasets/ML/Bridges_or_cracks/multiclassifier/Splits/Test/swin/without/ --image_size=299 --eval_split=test --resnet_depth=50   --checkpoint=/media/grvc/MAMG_ViGUS/GRVC/Parameters/Paper/Simclr/150_2/ckpt-numero --model_dir=/media/grvc/MAMG_ViGUS/GRVC/Parameters/Paper/Simclr/150_2 --use_tpu=False --eval_split val
```
