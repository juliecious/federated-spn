import torch
import pyjuice as juice
import pyjuice.nodes.distributions as juice_dists
from pyjuice.structures import PD, RAT_SPN
from torchvision.datasets import ImageNet, SVHN, CelebA
from torchvision.transforms import Compose, CenterCrop, Resize, ToTensor
from torch.utils.data import Subset, DataLoader
from rtpt import RTPT
import numpy as np

batch_size = 256

def bits_per_dim(nll, num_features):
    # If NLL is summed across a batch, take the mean
    nll_mean = nll.mean() if isinstance(nll, torch.Tensor) else np.mean(nll)
    
    # Compute bits per dimension
    bpd = nll_mean / (num_features * np.log(2))
    return bpd

def load_dataset(ds_name, split='train'):
    if ds_name == 'imagenet':
        transform = Compose([ToTensor(), Resize(16, antialias=True), CenterCrop(16)])
        dataset = ImageNet('/storage-01/datasets/imagenet/', transform=transform, split=split)
        data_shape = (16, 16, 3)
    elif ds_name == 'imagenet32':
        transform = Compose([ToTensor(), Resize(16, antialias=True), CenterCrop(16)])
        dataset = ImageNet('/storage-01/datasets/imagenet/', transform=transform, split=split)
        data_shape = (16, 16, 3)
    elif ds_name == 'celeba':
        transform = Compose([ToTensor(), Resize(16, antialias=True), CenterCrop(16)])
        dataset = CelebA('/storage-01/datasets/', transform=transform, split=split)
        data_shape = (16, 16, 3)
    return dataset, data_shape

def train(ds_name, num_epochs):

    rt = RTPT('JS', 'PyJuice', num_epochs)
    rt.start()
    device = torch.device(f'cuda:{6}')
    dataset, shape = load_dataset(ds_name)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    #arch = RAT_SPN(np.prod(list(shape)), 256, 2, 6, input_node_type=juice_dists.Categorical, input_node_params={'num_cats': 256})
    #input_dist = juice_dists.Gaussian(0.0, 1.0, 0.1)
    input_dist = juice_dists.Categorical(256)
    arch = PD(shape, 256, input_dist=input_dist, split_intervals=2)
    print(arch)
    model = juice.compile(arch)
    torch.cuda.set_device(device)
    model = model.to(device)

    for e in range(num_epochs):

        total_ll = 0.0

        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            x *= 256
            x = x.reshape(x.shape[0], -1).to(torch.int32)

            # This is equivalent to zeroing out the parameter gradients of a neural network
            model.init_param_flows(flows_memory = 0.0)
            # Forward pass
            lls = model(x)
            # Backward pass
            lls.mean().backward()
            total_ll += lls.sum().detach().cpu() / (len(loader) * loader.batch_size)
            # Mini-batch EM
            model.mini_batch_em(step_size = 0.02, pseudocount = 0.001)

            if i % 20 == 0:
                print(f"Epoch {e+1}/{num_epochs}: \t Iter: {i}/{len(loader)}: \t LL: {total_ll}")
        
        print(f"Epoch {e+1}/{num_epochs} \t LL: {total_ll}")
        rt.step()
    
    return model, device


def evaluate(model, dataset, device):

    loader = DataLoader(dataset, batch_size=256, num_workers=0)

    total_ll = 0.0
    lls_collect = []

    for x, y in loader:
        x = x.reshape(x.shape[0], -1)
        x = x.to(device)
        
        lls = model(x)
        lls_collect.append(lls.detach().cpu().numpy())
        total_ll += lls.sum().detach().cpu()

    return np.concatenate(lls_collect), total_ll / (len(loader) * loader.batch_size)


model, device = train('celeba', 10)
test_set, shape = load_dataset('celeba', 'valid')
lls, ll = evaluate(model, test_set, device)
bpd = bits_per_dim(-lls, np.prod(shape))
print(lls)
print(f"LL: {lls.mean()}")
print(f"BPD: {bpd}")
