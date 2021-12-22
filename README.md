# A Simple Baseline for Low-Budget Active Learning
[Kossar Pourahmadi](https://arghavan-kpm.github.io/),
[Parsa Nooralinejad](https://p-nooralinejad.github.io/),
[Hamed Pirsiavash](https://www.csee.umbc.edu/~hpirsiav/)<br/>

This repository is the implementation of [A Simple Baseline for Low-Budget Active Learning](https://arxiv.org/abs/2110.12033).

<p align="center">
  <img src="https://github.com/UCDvision/low-budget-al/blob/main/docs/assets/images/teaser.PNG" width="85%">
</p>


In this paper, we are interested in low-budget active learning where only a small subset of unlabeled data, e.g. 0.2% of ImageNet, can be annotated. Instead of proposing a new query strategy to iteratively sample batches of unlabeled data given an initial pool, we learn rich features by an off-the-shelf self-supervised learning method only once and then study the effectiveness of different sampling strategies given a low budget on a variety of datasets as well as ImageNet dataset. We show that although the state-of-the-art active learning methods work well given a large budget of data labeling, a simple k-means clustering algorithm can outperform them on low budgets. We believe this method can be used as a simple baseline for low-budget active learning on image classification. Our code is modified from [CompRess](https://github.com/UMBCvision/CompRess) [1]. 

```
@article{pourahmadi2021simple,
  title={A Simple Baseline for Low-Budget Active Learning},
  author={Pourahmadi, Kossar and Nooralinejad, Parsa and Pirsiavash, Hamed},
  journal={arXiv preprint arXiv:2110.12033},
  year={2021}
}
```

# Table of Content

1. [Benchmarks](#Benchmarks)
2. [Requirements](#Requirements)
3. [Evaluation](#Evaluation)
	1. [Sample selection](#sample_selection)
	2. [Linear classification](#linear)
	3. [Nearest neighbor classification](#nn)
	4. [Max-Entropy sampling](#entropy)
	5. [Fine-tuning](#ft)
	6. [Training from scratch](#from_scratch)
4. [References](#ref)
    
## Benchmarks <a name="Benchmarks"></a>

We implemented the following query strategies in ```strategies.py``` on **CIFAR-10**, **CIFAR-100**, **ImageNet**, and **ImageNet-LT** datasets:

**a) Single-batch k-means:** At each round, it clusters the whole dataset to budget size clusters and sends nearest neighbors of centers directly to the oracle to be annotated.

**b) Multi-batch k-means:** Uses the difference of two consecutive budget sizes as the number of clusters and picks those nearest examples to centers that have not been labeled previously by the oracle.

**c) Core-set [2]**

**d) Max-Entropy [3]:** Treats the entropy of example probability distribution output as an uncertainty score and samples uncertain points for annotation.

**e) Uniform:** Selects equal number of samples randomly from all classes.

**f) Random:** Samples are selected randomly (uniformly) from the entire dataset.

## Requirements <a name="Requirements"></a>

* Python 3.7
* [PyTorch](https://pytorch.org/)
* ImageNet dataset: Follow the instructions at [official ImageNet training in PyTorch repo](https://github.com/pytorch/examples/tree/master/imagenet) to setup.
* [FAISS](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md): To perform k-means and nearest neighbor classification, we use FAISS GPU library.
* Download ImageNet_LT_train.txt from [here](https://drive.google.com/drive/u/1/folders/19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_-) and put it in folder ```data/```.

## Evaluation <a name="Evaluation"></a>

This implementation supports multi-gpu, DataParallel or single-gpu training. 

You have the following options to run commands:

* ```--arch``` We use pre-trained ResNet-18 with CompRess [(download weights)](https://drive.google.com/file/d/1L-RCmD4gMeicxJhIeqNKU09_sH8R3bwS/view?usp=sharing) or pre-trained ResNet-50 with MoCo-v2 [(download weights)](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar). Use one of ```resnet18``` or ```resnet50``` as the argument accordingly.
* ```--backbone``` compress, moco
* ```--splits``` You can define budget sizes with comma as a seperator. For instance, ```--splits 10,20```.
* ```--name``` Specify the query strategy name by using one of ```uniform random kmeans accu_kmeans coreset```.
* ```--dataset``` Indicate the unlabeled dataset name by using one of ```cifar10 cifar100 imagenet imagenet_lt```.

### Sample selection <a name="sample_selection"></a>
If the strategy needs an initial pool (accu_kmeans or coreset) then pass the file path with ```--resume-indices```.

```bash
python sampler.py \
--arch resnet18 \
--weights [path to weights] \
--backbone compress \
--batch-size 4 \
--workers 4 \
--splits 100 \
--load_cache \
--name kmeans \
--dataset cifar10 \
[path to dataset file]
```
Category coverage results of selected samples on ImageNet:
| Method | 0.08% | 0.2% |  0.5% | ≥ 1% |
|--------|-------|------|-------|------|
|Uniform |100 | 100 | 100 | 100|
|Random | 62.9 | 94.6 | 100 | 100|
|Max-Entropy | 62.9 | 84.3 | 94.8 | 100|
|Core-set | 62.9 | 87.9 | 97.0 | 100|
|VAAL | 62.9 | 94.6 | 98.1 | 100|
|Multi k-means | 72.2 | 97.0 | 99.8 | 100|
|K-means | 72.2 | 97.8 | 99.9 | 100|

### Linear classification <a name="linear"></a>

```bash
python eval_lincls.py \
--arch resnet18 \
--weights [path to weights] \
--backbone compress \
--batch-size 128 \
--workers 4 \
--lr 0.01 \
--lr_schedule 50,75 \
--epochs 100 \
--splits 1000 \
--load_cache \
--name random \
--dataset imagenet \
[path to dataset file]
```
Top-1 linear classification results on ImageNet:

| Method | 0.08%(1K) | 0.2%(3K) |  0.5%(7K) | 1%(13K) | 2%(26K) | 5%(64K) | 10%(128K) | 15%(192K) |
|--------|-----------|----------|-----------|---------|---------|---------|-----------|-----------|
|Uniform | 19.2  |31.9 |41.0 |46.0  |49.9  |**54.2** | **56.7** | 57.9 |
|Random  |15.8   |28.0  |39.2  |45.1  |49.7 |54.0  |56.6   |57.9 |
|Max-Entropy |15.8 |19.4  |25.6  |33.7 |41.3  |48.9  |51.9 |54.3 |
|Core-set |15.8  |25.6  |33.3  |39.6  |45.7 |51.3  |54.9  |56.6 |
|VAAL |15.8  |27.7  |34.9  |42.8  |49.2 |53.6 |56.0 |57.4|
|Multi k-means | **24.6**  |34.1  |41.1 |45.3  |49.5  |53.9 |56.3 |57.5 |
|K-means | **24.6** | **35.7** | **42.6** | **46.9**| **50.7** | 54.0 | 56.6 | **58.0** |

### Nearest neighbor classification <a name="nn"></a>

```bash
python eval_knn.py \
--arch resnet18 \
--weights [path to weights] \
--backbone compress \
--batch-size 128 \
--workers 8 \
--splits 1000 \
--load_cache \
--name random \
--dataset cifar10 \
[path to dataset file]
```
1-NN results on ImageNet:

| Method | 0.08%(1K) | 0.2%(3K) |  0.5%(7K) | 1%(13K) | 2%(26K) | 5%(64K) | 10%(128K) | 15%(192K) |
|--------|-----------|----------|-----------|---------|---------|---------|-----------|-----------|
|Uniform | 29.5 | 35.7 | 38.9 | 41.1 | 43.2 | 45.6 | 47.6 | 48.6|
|Random | 22.8 | 33.2 | 38.4 | 40.8 | 42.2 | 45.4 | 47.3 | 48.3|
|Max-Entropy | 22.8 | 24.5 | 27.2 | 30.3 | 33.3 | 36.2 | 37.6 | 38.6 |
|Core-set | 22.8 | 30.7 | 34.8 | 37.5 | 39.7 | 42.0 | 43.7 | 44.6|
|VAAL | 22.8 | 32.8 | 36.2 | 39.7 | 42.6 | 45.3 | 46.7 | 47.9|
|Multi k-means | **31.6** | 38.2 | 41.4 | 43.3 | 45.2 | 47.2 | **48.6** |  **49.4**|
|K-means | **31.6** | **39.9** | **42.7** | **44.0** | **45.5** | **46.8** | 48.1 | 48.8|


### Max-Entropy sampling <a name="entropy"></a>
To sample data using Max-Entropy, use ```active_sampler.py``` and ```entropy``` for ```--name```. Give the initial pool indices file path with --resume-indices.

```bash
python active_sampler.py \
--arch resnet18 \
--weights [path to weights] \
--backbone compress \
--batch-size 128 \
--workers 4 \
--lr 0.001 \
--lr_schedule 50,75 \
--epochs 100 \
--splits 2000 \
--load_cache \
--name entropy \
--resume-indices [path to random initial pool file] \
--dataset imagenet \
[path to dataset file]
```

### Fine-tuning <a name="ft"></a>
This file is implemented only for CompRess ResNet-18 backbone on **ImageNet**. ```--lr``` is the learning rate of backbone and ```--lr-lin``` is for the linear classifier.
```bash
python finetune.py \
--arch resnet18 \
--weights [path to weights] \
--batch-size 128 \
--workers 16 \
--epochs 100 \
--lr_schedule 50,75 \
--lr 0.0001 \
--lr-lin 0.01 \
--splits 1000 \
--name kmeans \
--dataset imagenet \
[path to dataset file]
```

### Training from scratch <a name="from_scratch"></a>
Starting from a random initialized network, you can train the model on **CIFAR-100** or **ImageNet**.
```bash
python trainer_DP.py \
--arch resnet18 \
--batch-size 128 \
--workers 4 \
--epochs 100 \
--lr 0.1 \
--lr_schedule 30,60,90 \
--splits 1000 \
--name kmeans \
--dataset imagenet \
[path to dataset file]
```

## References <a name="ref"></a>

[1] CompRess: Self-Supervised Learning by Compressing Representations, NeurIPS, 2020

[2] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[3] A new active labeling method for deep learning, IJCNN, 2014
