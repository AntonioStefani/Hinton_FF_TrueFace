import os
import random
from datetime import timedelta

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from src import ff_mnist, ff_model, ff_trueface, ff_vit_model


def parse_args(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    random.seed(opt.seed)

    print(OmegaConf.to_yaml(opt))
    return opt


def get_model_and_optimizer(opt):
    if opt.model.type == "MLP":
        model = ff_model.FF_model(opt)
    elif opt.model.type == "ViT":
        model = ff_vit_model.FF_ViT_model(opt)
    if "cuda" in opt.device:
        model = model.to(device=opt.device)
    print(model, "\n")

    # Create optimizer with different hyper-parameters for the main model
    # and the downstream classification model.
    main_model_params = [
        p
        for p in model.parameters()
        if all(p is not x for x in model.linear_classifier.parameters())
        if all(p is not x for x in model.linear_classifier.parameters())
    ]
    optimizer = torch.optim.SGD(
        [
            {
                "params": main_model_params,
                "lr": opt.training.learning_rate,
                "weight_decay": opt.training.weight_decay,
                "momentum": opt.training.momentum,
            },
            {
                "params": model.linear_classifier.parameters(),
                "params": model.linear_classifier.parameters(),
                "lr": opt.training.downstream_learning_rate,
                "weight_decay": opt.training.downstream_weight_decay,
                "momentum": opt.training.momentum,
            },
        ]
    )
    return model, optimizer


def get_data(opt, partition):

    if opt.input.dataset == "TrueFace":
        dataset = ff_trueface.FF_TrueFace(opt)
        dset = get_TrueFace_partition(dataset, partition)

    elif opt.input.dataset == "MNIST":
        dset = ff_mnist.FF_MNIST(opt, partition)

    # Improve reproducibility in dataloader.
    g = torch.Generator()
    g.manual_seed(opt.seed)

    return torch.utils.data.DataLoader(
        dset,
        batch_size=opt.input.batch_size,
        drop_last=True,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
        num_workers=4,
        persistent_workers=True,
    )


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_TrueFace_partition(dataset, partition):
    if partition in ["train", "val"]:
        train_length = int(len(dataset)*0.9)
        train_set, validation_set = random_split(dataset, [train_length, len(dataset) - train_length])

        if partition == "train":
            dset = train_set
        elif partition == "val":
            dset = validation_set
        if partition == "train":
            dset = train_set
        elif partition == "val":
            dset = validation_set
    elif partition == "test":
        dset = dataset
        dset = dataset
    else:
        raise NotImplementedError
    
    return dset


def get_MNIST_partition(opt, partition):
    if partition in ["train", "val", "train_val"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    elif partition in ["test"]:
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
    else:
        raise NotImplementedError

    if partition == "train":
        mnist = torch.utils.data.Subset(mnist, range(50000))
    elif partition == "val":
        mnist = torchvision.datasets.MNIST(
            os.path.join(get_original_cwd(), opt.input.path),
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        mnist = torch.utils.data.Subset(mnist, range(50000, 60000))

    return mnist


def dict_to_cuda(opt, dict):
    for key, value in dict.items():
        dict[key] = value.to(device=opt.device, non_blocking=True)
    return dict


def preprocess_inputs(opt, inputs, labels):
    if "cuda" in opt.device:
        inputs = dict_to_cuda(opt, inputs)
        labels = dict_to_cuda(opt, labels)
    return inputs, labels


def get_linear_cooldown_lr(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * 2 * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def get_linear_cooldown_lr_warmup(opt, epoch, lr):
    if epoch > (opt.training.epochs // 2):
        return lr * (opt.training.epochs/(opt.training.epochs - opt.training.warmup)) * (1 + opt.training.epochs - epoch) / opt.training.epochs
    else:
        return lr


def update_learning_rate(optimizer, opt, epoch):
    optimizer.param_groups[0]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.learning_rate
    )
    optimizer.param_groups[1]["lr"] = get_linear_cooldown_lr(
        opt, epoch, opt.training.downstream_learning_rate
    )
    return optimizer


def get_accuracy(opt, output, target):
    """Computes the accuracy."""
    with torch.no_grad():
        prediction = torch.argmax(output, dim=1)
        return (prediction == target).sum() / opt.input.batch_size


def print_results(partition, iteration_time, scalar_outputs, epoch=None):
    if epoch is not None:
        print(f"Epoch {epoch} \t", end="")

    print(
        f"{partition} \t \t"
        f"Time: {timedelta(seconds=iteration_time)} \t",
        end="",
    )
    if scalar_outputs is not None:
        for key, value in scalar_outputs.items():
            print(f"{key}: {value:.4f} \t", end="")
    print()


def log_results(result_dict, scalar_outputs, num_steps):
    for key, value in scalar_outputs.items():
        if isinstance(value, float):
            result_dict[key] += value / num_steps
        else:
            result_dict[key] += value.item() / num_steps
    return result_dict
