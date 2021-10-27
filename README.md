# A Simple Baseline for Low-Budget Active Learning

This repository is the implementation of [A Simple Baseline for Low-Budget Active Learning](). In this paper, we are interested in low-budget active learning where only a small subset of unlabeled data, e.g. 0.2% of ImageNet, can be annotated. We show that although the state-of-the-art active learning methods work well given a large budget of data labeling, a simple k-means clustering algorithm can outperform them on low budgets. Our code is modified from [CompRess](https://github.com/UMBCvision/CompRess) [1]. 

```
@misc{pourahmadi2021simple,
      title={A Simple Baseline for Low-Budget Active Learning}, 
      author={Kossar Pourahmadi and Parsa Nooralinejad and Hamed Pirsiavash},
      year={2021},
      eprint={2110.12033},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Benchmarks

We implemented the following query strategies in ```strategies.py``` on **CIFAR-10**, **CIFAR-100**, **ImageNet**, and **ImageNet-LT** datasets:

**a) K-means:** At each round, it clusters the whole dataset to budget size clusters and sends nearest neighbors of centers directly to the oracle to be annotated.

**b) Accumulative k-means:** Uses the difference of two consecutive budget sizes as the number of clusters and picks those nearest examples to centers that have not been labeled previously by the oracle.

**c) Coreset [2]**

**d) Max-Entropy [3]:** Treats the entropy of example probability distribution output as an uncertainty score and samples uncertain points for annotation.

**e) Uniform:** Selects equal number of samples randomly from all classes.

**f) Random:** Samples are selected randomly (uniformly) from the entire dataset.

## Requirements

* Python 3.7
* [PyTorch](https://pytorch.org/)
* ImageNet dataset: Follow the instructions at [official ImageNet training in PyTorch repo](https://github.com/pytorch/examples/tree/master/imagenet) to setup.
* [FAISS](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md): To perform k-means and nearest neighbor classification, we use FAISS GPU library.
* Download ImageNet_LT_train.txt from [here](https://drive.google.com/drive/u/1/folders/19cl6GK5B3p5CxzVBy5i4cWSmBy9-rT_-) and put it in folder ```data/```.

## Usage

This implementation supports multi-gpu, DataParallel or single-gpu training. 

You have the following options to run commands:

* ```--arch``` We use pre-trained ResNet-18 with CompRess [(download weights)](https://drive.google.com/file/d/1L-RCmD4gMeicxJhIeqNKU09_sH8R3bwS/view?usp=sharing) or pre-trained ResNet-50 with MoCo-v2 [(download weights)](https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar). Use one of ```resnet18``` or ```resnet50``` as the argument accordingly.
* ```--backbone``` compress, moco
* ```--splits``` You can define budget sizes with comma as a seperator. For instance, ```--splits 10,20```.
* ```--name``` Specify the query strategy name by using one of ```uniform random kmeans accu_kmeans coreset```.
* ```--dataset``` Indicate the unlabeled dataset name by using one of ```cifar10 cifar100 imagenet imagenet_lt```.

### Sample selection
If the strategy needs an initial pool (accu_kmeans or coreset) then pass the file path with ```--resume-indices```.

```
python sampler.py --arch resnet18 \
--weights [path to weights] --backbone compress \
--batch-size 4 --workers 4 \
--splits 100  \
--load_cache --name kmeans \
--dataset cifar10 [path to dataset file]
```

### Linear classification

```
python eval_lincls.py --arch resnet18 \
--weights [path to weights] --backbone compress \
--batch-size 128 --workers 4 --lr 0.01 --lr_schedule 50,75 --epochs 100 \
--splits 1000 \  
--load_cache --name random \
--dataset imagenet [path to dataset file]
```

### Nearest neighbor classification

```
python eval_knn.py --arch resnet18 \
--weights [path to weights] --backbone compress \
--batch-size 128 --workers 8 \
--splits 1000 \
--load_cache --name random \
--dataset cifar10 [path to dataset file]
```

### Entropy sampling
To sample data using Max-Entropy, use ```active_sampler.py``` and ```entropy``` for ```--name```. Give the initial pool indices file path with --resume-indices.

```
python active_sampler.py --arch resnet18 \
--weights [path to weights] --backbone compress \
--batch-size 128 --workers 4 --lr 0.001 --lr_schedule 50,75 --epochs 100 \
--splits 2000 \
--load_cache --name entropy \
--resume-indices [path to random initial pool file] \
--dataset imagenet [path to dataset file]
```

### Fine-tuning
This file is implemented only for CompRess ResNet-18 backbone on **ImageNet**. ```--lr``` is the learning rate of backbone and ```--lr-lin``` is for the linear classifier.
```
python finetune.py --arch resnet18 \
--weights [path to weights] \
--batch-size 128 --workers 16 --epochs 100 --lr_schedule 50,75 \
--lr 0.0001 --lr-lin 0.01 \
--splits 1000 --name kmeans \
--dataset imagenet [path to dataset file]
```

### Training from scratch
Starting from a random initialized network, you can train the model on **CIFAR-100** or **ImageNet**.
```
python trainer_DP.py --arch resnet18 \
--batch-size 128 --workers 4 --epochs 100 --lr 0.1 --lr_schedule 30,60,90 \
--splits 1000 --name kmeans \
--dataset imagenet [path to dataset file]
```

## References

[1] CompRess: Self-Supervised Learning by Compressing Representations, NeurIPS, 2020

[2] Active Learning for Convolutional Neural Networks: A Core-Set Approach, ICLR, 2018

[3] A new active labeling method for deep learning, IJCNN, 2014
