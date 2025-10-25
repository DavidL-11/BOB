## D-FINE
D-FINE is a powerful real-time object detector that redefines the bounding box regression task in DETRs as Fine-grained Distribution Refinement (FDR) and introduces Global Optimal Localization Self-Distillation (GO-LSD), achieving outstanding performance without introducing additional inference and training costs.



## Training
For training the D-FINE model, you need to have the dataset prepared in the COCO format.
To achieve this on the existing datasets, run the `convert_datasets.py` script and the 'yolo_to_coco_label.py' script to convert the dataset annotations to the COCO format afterwards.

Requirements for training:

- SegFM repository cloned and installed
- BOB installed
- torch>=2.0.1
- torchvision>=0.15.2
- faster-coco-eval>=1.6.6
- PyYAML
- tensorboard
- scipy
- calflops
- transformers
- loguru

To train the model, activate your conda environment for D-FINE and run the following command from the root directory of the project:

```bash
python src/segFM/BOB/src/BOB/train/dfine/model/train.py -c src/segFM/BOB/src/BOB/train/dfine/dfine_hgnetv2_n_custom.yml --use-amp --seed=0 --output-dir="runs/detect/BOB_dfine"
```

To visualize the training process, you can use:

```bash
tensorboard --logdir ./runs/detect/BOB_dfine
```

### Modyfing Batch Size and Epochs
If you want to modify the batch size, you can do so in the 'src/segFM/BOB/models/dfine/dfine_hgnetv2_n_custom.yml' file under the 'train_dataloader' section:

```yaml
epochs: 150 # Change this value to your desired number of epochs
train_dataloader:
  total_batch_size: 32 # Change this value to your desired batch size
  dataset:
    transforms:
      policy:
        epoch: 70 # This value sets when the data augmentation policy will be applied, i.e., after how many epochs the transforms should stop being applied
  collate_fn:
    stop_epoch: 70 # This value sets when the BatchImageCollateFunction will stop applying the transforms, usually matching the 'epoch' value above
```


### Modifying Input Image Size
If you want to modify the input image size (default of D-FINE is 640x640), you need to change 

In 'model/configs/dfine/include/dataloader.yml':

- the scaling parameters '- {type: Resize, size: [512, 512], }' in the train and val dataload 
- 'base_size: 512' in the 'collate_fn' section.

In 'model/configs/dfine/include/dfine_hgnetv2.yml':

- 'eval_spatial_size: [512, 512] # h w'

## Inference
To perform inference with the trained model, you can use the dfine_inference.py script

## Exporting to ONNX
To export the trained model to ONNX format, you can use the `export_onnx.py` script.

```sh
python src/segFM/BOB/src/BOB/train/dfine/model/tools/deployment/export_onnx.py --check -c src/segFM/BOB/src/BOB/train/dfine/dfine_hgnetv2_n_custom.yml -r runs/detect/BOB_dfine/best_stg2.pth
```

Be sure that `cmake` is installed on your system and `onnx` and `onnxruntime` are installed in your Python environment.

If you are exporting a model with a different input size than 640x640, make sure to modify the `data` and `size` variables in the `export_onnx.py` script accordingly. The `onnx_inf` script additionally
requires the input size to be modified (`resize_with_aspect_ratio(im_pil, 512)`), but you most likely don't need it, as it is included in the `dfine_inference.py` script.