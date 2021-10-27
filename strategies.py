from tqdm import tqdm
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def core_set(unlabeled_emb, labeled_emb, unlabeled_indices, budget, args):
    candidates = np.ones(budget, dtype=np.int32)*-1

    unlabeled_emb = unlabeled_emb.numpy()

    labeled_emb = labeled_emb.numpy()

    n, dim = unlabeled_emb.shape

    # find nearest neighbor distances to each unlabeled data
    index = faiss.IndexFlatL2(dim)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(labeled_emb)
    min_distance, _ = gpu_index.search(unlabeled_emb, 1)
    min_distance = np.array(min_distance[:, 0])

    print("select candidates =>")
    for i in tqdm(range(budget)):
        candidate = min_distance.argmax()
        assert unlabeled_indices[candidate] not in candidates, "duplicate selection"
        candidates[i] = unlabeled_indices[candidate]
        
        gpu_index = faiss.index_cpu_to_all_gpus(index, co)
        gpu_index.add(unlabeled_emb[candidate].reshape(1, -1))
        new_min_distance, _ = gpu_index.search(unlabeled_emb, 1)
        new_min_distance = np.array(new_min_distance[:, 0])

        for j in range(n):
            min_distance[j] = min(min_distance[j], new_min_distance[j])
        assert '%0.2f' % min_distance[candidate] == '0.00', '%0.2f' % min_distance[candidate]

    return candidates


def accu_kmeans(inference_feats, budget, unlabeled_indices, args):
    clustermembers = []

    inference_feats = inference_feats.numpy()

    n, dim = inference_feats.shape
    max_points_per_centroid = int(2*args.num_images // budget)

    kmeans = faiss.Kmeans(dim, budget, niter=20, nredo=5, verbose=True, max_points_per_centroid=max_points_per_centroid, gpu=True)
    kmeans.train(inference_feats)   
    centroids = kmeans.centroids

    index = faiss.IndexFlatL2(dim)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(centroids)
    _, assignments = gpu_index.search(inference_feats, 1)
    assignments = np.array(assignments[:, 0])

    clustermembers.extend(np.where(assignments==i)[0] for i in range(budget))

    # find k nearest neighbors to each centroid
    index = faiss.IndexFlatL2(dim)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(inference_feats)
    k = 200
    _, nn = gpu_index.search(centroids, k)

    empty_clusters = 0
    valid_nn = np.ones(budget, dtype=np.int32)*-1
    for i in tqdm(range(budget), desc="valid nn:"):
        valid_members = np.array(nn[i])[np.in1d(np.array(nn[i]), np.intersect1d(unlabeled_indices, np.array(clustermembers[i])))]
        if len(valid_members):
            valid_nn[i] = valid_members[0]

        if not len(np.intersect1d(unlabeled_indices, np.array(clustermembers[i]))):
            empty_clusters += 1

    assert len(np.where(valid_nn==-1)[0]) == empty_clusters, f'knn is small: {empty_clusters} empty clusters, {len(np.where(valid_nn==-1)[0])} clusters without nn'
    valid_nn = np.delete(valid_nn, np.where(valid_nn == -1)[0])

    return valid_nn

def fast_kmeans(inference_feats, budget, args):

    clustermembers = []
    candidates = np.ones(budget, dtype=np.int32)*-1

    inference_feats = inference_feats.numpy()

    n, dim = inference_feats.shape
    max_points_per_centroid = int(2*args.num_images // budget)

    kmeans = faiss.Kmeans(dim, budget, niter=20, nredo=5, verbose=True, max_points_per_centroid=max_points_per_centroid, gpu=True)
    kmeans.train(inference_feats)   
    centroids = kmeans.centroids

    index = faiss.IndexFlatL2(dim)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(centroids)
    _, assignments = gpu_index.search(inference_feats, 1)
    assignments = np.array(assignments[:, 0])

    clustermembers.extend(np.where(assignments==i)[0] for i in range(budget))

    # find k nearest neighbors to each centroid
    index = faiss.IndexFlatL2(dim)
    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = True
    gpu_index = faiss.index_cpu_to_all_gpus(index, co)
    gpu_index.add(inference_feats)
    k = 200
    _, nn = gpu_index.search(centroids, k)

    empty_clusters = 0
    valid_nn = np.ones(budget, dtype=np.int32)*-1
    for i in tqdm(range(budget), desc="valid nn:"):
        valid_members = np.array(nn[i])[np.in1d(np.array(nn[i]), np.array(clustermembers[i]))]
        if len(valid_members):
            valid_nn[i] = valid_members[0]

        if not len(np.array(clustermembers[i])):
            empty_clusters += 1

    assert len(np.where(valid_nn==-1)[0]) == empty_clusters, f'knn is small: {empty_clusters} empty clusters, {len(np.where(valid_nn==-1)[0])} clusters without nn'
    valid_nn = np.delete(valid_nn, np.where(valid_nn == -1)[0])
    
    return valid_nn

def uniform(inference_labels, splits, args):
    candidates = []
    for i in range(args.num_classes):
        candidates.append(np.random.choice(np.where(inference_labels==i)[0], len(np.where(inference_labels==i)[0]), replace=False))

    splits = [i // args.num_classes for i in splits]

    for split in splits:
        samples = np.zeros(split*args.num_classes, dtype=np.int32) * -1
        for i in range(args.num_classes):
            # this is for imagenet-lt that some classes have few examples
            ptr = min(split, len(np.where(inference_labels==i)[0]))
            samples[i*split:i*split+ptr] = candidates[i][:split]

        samples = np.delete(samples, np.where(samples==-1)[0])

        print('{} inidices are sampled in total, {} of them are unique'.format(len(samples), len(np.unique(samples))))
        print('{}% of all categories is seen'.format(len(np.unique(inference_labels[samples]))/args.num_classes * 100))
        np.save(f'{args.inidices}/uniform_{args.dataset}_{split*args.num_classes}.npy', samples)
    
def random(all_indices, inference_labels, splits, args):
    shuffled_indices = np.random.choice(all_indices, len(all_indices), replace=False)
    for split in splits:
        samples = np.zeros(split, dtype=np.int32)
        samples = shuffled_indices[:split]
        print('{} inidices are sampled in total, {} of them are unique'.format(len(samples), len(np.unique(samples))))
        print('{}% of all categories is seen'.format(len(np.unique(inference_labels[samples]))/args.num_classes * 100))
        np.save(f'{args.indices}/random_{args.dataset}_{split}.npy', samples)

def entropy(unlabeled_loader, unlabeled_indices, backbone, linear, budget):

    backbone.eval()
    linear.eval()
    logits, ptr = None, 0

    with torch.no_grad():
        for images, _, _ in tqdm(unlabeled_loader, desc="unlabeled features extraction"):

            images = images.cuda(non_blocking=True)

            # compute output
            cur_logits = linear(backbone(images)).cpu()
            B, D = cur_logits.shape
            inds = torch.arange(B) + ptr

            if not ptr:
                logits = torch.zeros((len(unlabeled_loader.dataset), D)).float()

            logits.index_copy_(0, inds, cur_logits)
            ptr += B

    probs = F.softmax(logits, dim=1)

    entropies = torch.sum(probs * torch.log(probs), dim=1)

    return unlabeled_indices[entropies.sort()[1][:budget]]


def normalize(x):
    return x / x.norm(2, dim=1, keepdim=True)