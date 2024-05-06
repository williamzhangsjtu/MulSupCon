# 【AAAI'2024】Multi-Label Supervised Contrastive Learning (MulSupCon)

Official implement of [Multi-Label Supervised Contrastive Learning](https://ojs.aaai.org/index.php/AAAI/article/view/29619). We introduces a novel contrastive loss function, termed “MulSupCon”, which effectively extends the single-label supervised contrastive learning to the multi-label context.

## Data Preparation

We use hdf5 file to store data.

### Vector dataset
The file contains *feature* and *target* dataset
- feature: A numpy array of input data which has shape (B, D), where B represents data size and D is the input dimension.
- target: A numpy array of label which has shape (B, C), where C is the number of class

### Image dataset
The file contains *Image* and *target* dataset
- Image: A numpy array of input data which has shape (B, H, W, 3), where B represents data size.
- target: A numpy array of label which has shape (B, C), where C is the number of class

The data files are located in the `data/` directory.

## Configuration

The configuration settings are specified in the config file in `codes/config/` directory.

## Pretrain

We give an example of pretraining on yeast dataset. The first step is to enter the `codes/` directory and modify configuration in `config/pretrain_yeast.yaml` if necessary. Then pretrain with the following codes:

```python
python main.py pretrain \
--config config/pretrain_yeast.yaml \
--is_image False # False for vector data and True for image data
```

By default the pretrained model will be stored in `experiment/yeast` directory

## Classification with pretrained model

Then we can conduct multi-label classification using pretrained model. The first step is to enter the `codes/` directory and modify configuration in `config/train_yeast.yaml` if necessary. Then train with the following codes:


```python
python main.py train \
--config config/train_yeast.yaml \
--is_image False # False for vector data and True for image data
--model_path <model_dir> # The directory which contains .pt checkpoint and configuration file.
--linear_probe False # Finetune the whole model
```

## 📚Citation
```
@inproceedings{zhang2024multi,
  title={Multi-Label Supervised Contrastive Learning},
  author={Zhang, Pingyue and Wu, Mengyue},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={15},
  pages={16786--16793},
  year={2024}
}
```