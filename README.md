# IS-MIL Training & validation



## Prepare

### Step 1. prepare for data

Panda data can be download in https://www.kaggle.com/competitions/prostate-cancer-grade-assessment/data

DiagSet data can be download in https://github.com/michalkoziarski/DiagSet

You need to process the WSI into the following format. The processing method can be found in  https://github.com/mahmoodlab/CLAM

The **CTransPath** feature extractor and the pretrained model can be download in  https://github.com/Xiyue-Wang/TransPath

**In order to train ISMIL, you need to prepare two folders, one with no overlap between patches and the other with an overlap of 0.5**

```
DATA_DIR
├─patch_coord
│      slide_id_1.h5
│      slide_id_2.h5
│      ...
└─patch_feature
        slide_id_1.pt
        slide_id_2.pt
        ...
```

The h5 file in the `patch_coord` folder contains the coordinates of each patch of the WSI, which can be read as

```python
coords = h5py.File(coords_path, 'r')['coords'][:]
# coords is a array like:
# [[x1, y1], [x2, y2], ...]
```

The pt file in the `patch_feature`folder contains the features of each patch of the WSI, which can be read as

```python
features = torch.load(features_path, map_location=torch.device('cpu'))
# features is a tensor with dimension N*F, and if features are extracted using CTransPath, F is 768
```

### Step 2. preparing the data set split

You need to divide the dataset into a training set validation set and a test set, and store them in the following format

```
SPLIT_DIR
    test_set.csv
    train_set.csv
    val_set.csv
```

And, the format of the csv file is as follows

| slide_id   | label |
| ---------- | ----- |
| slide_id_1 | 0     |
| slide_id_2 | 1     |
| ...        | ...   |

## Train model

### Step 1. create a config file

We have prepared two config file templates (see ./configs/) for ISMIL and other baselines, like

```yaml
General:
    seed: 7
    work_dir: WORK_DIR
    fold_num: 4

Data:
    split_dir: SPLIT_DIR
    data_dir_1: DATA_DIR_1 # for data with no overlap
    data_dir_2: DATA_DIR_2 # for data with overlap
    features_size: 768
    n_classes: 2

Model:
    network: 'ISMIL'

Train:
    mode: repeat
    lr: 3.0e-4
    reg: 1.0e-5
    CosineAnnealingLR:
        T_max: 30
        eta_min: 1.0e-7
    Early_stopping:
        patient: 10
        stop_epoch: 30
        type: max
    max_epochs: 100
    train_method: ISMIL
    val_method: ISMIL
    dataset: TwoStreamBagDataset

Test:
    test_set_name:
        data_dir_1: TEST_DATA_DIR_1 # for data with no overlap
        data_dir_2: TEST_DATA_DIR_2 # for data with overlap
        csv_path: TEST_SET_CSV_PATH
```

In the config, the correspondence between the `Model.network`, `Train.training_method` and `Train.val_method` is as follows

| `Model.network` | `Train.training_method` | `Train.val_method` |
| --------------- | ----------------------- | ------------------ |
| ISMIL           | ISMIL                   | ISMIL              |

### Step 2. train model

Run the following command

```shell
python train.py --config_path [config path] --begin [begin index] --end [end index]
```

`--begin` and `--end` used to control repetitive experiments

When the training is finished, the code creates a directory with the same name as the config file in the working directory to store all the experimental data. Like this

```
Task_DIR
│  CONFIG_FILE.yaml
│  s_0_checkpoint.pt
│  s_1_checkpoint.pt
│  s_2_checkpoint.pt
│  s_3_checkpoint.pt
│  s_4_checkpoint.pt
│
└─results
    │
    └─test_set
            metrics.csv
            probs.csv
```



## Evaluation

Please add information about the test set in the config, like this

```yaml
Test:
    TEST_SET_NAME:
        data_dir_1: TEST_DATA_DIR_1 # for data with no overlap
        data_dir_2: TEST_DATA_DIR_2 # for data with overlap
        csv_path: TEST_SET_CSV_PATH
```

And run the command

```shell
python eval.py --config_path [config path] --dataset_name [TEST_SET_NAME]
```

Test result will save at `WORK_DIR/CONFIG_FILE_NAME/results/TEST_SET_NAME`

