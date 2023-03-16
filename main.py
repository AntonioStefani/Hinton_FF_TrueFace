import time
from collections import defaultdict

import wandb
from tqdm import tqdm

import hydra
import torch
from omegaconf import DictConfig

from src import utils
import os


def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for batch_idx, (inputs, labels) in enumerate(tepoch):
                inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

                optimizer.zero_grad()

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    scalar_outputs = model(inputs, labels)
                    # scalar_outputs["Loss"].backward()

                scaler.scale(scalar_outputs["Loss"]).backward()
                scaler.step(optimizer)
                scaler.update()

                # optimizer.step()

                train_results = utils.log_results(
                    train_results, scalar_outputs, num_steps_per_epoch            
                )
                
                if opt.wandb.enabled:
                    wandb.log({"train/loss": train_results["Loss"]},step=epoch)
                    wandb.log({"train/classification_loss": train_results["classification_loss"]},step=epoch)
                    wandb.log({"train/classification_accuracy": train_results["classification_accuracy"]},step=epoch)
                    for l in range(opt.model.num_layers):
                        wandb.log({f"train/loss_layer_{l}": train_results[f"loss_layer_{l}"]},step=epoch)
                tepoch.set_postfix(loss=train_results["Loss"], closs=train_results["classification_loss"], acc=train_results["classification_accuracy"])

            if not os.path.exists(f"checkpoint/{wandb.run.id}"):
                os.makedirs(f"checkpoint/{wandb.run.id}")
            torch.save(model.state_dict(), f"checkpoint/{wandb.run.id}/weights_{opt.input.pre_post}.pth")

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate_or_test(opt, model, "val", epoch=epoch)

    return model


def validate_or_test(opt, model, partition, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, partition)
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print(partition)
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

            if opt.wandb.enabled:
                wandb.log({str(partition)+"/loss": test_results["Loss"]},step=epoch)
                wandb.log({str(partition)+"/classification_loss": test_results["classification_loss"]},step=epoch)
                wandb.log({str(partition)+"/classification_accuracy": test_results["classification_accuracy"]},step=epoch)

    utils.print_results(partition, time.time() - test_time, test_results, epoch=epoch)
    model.train()


@hydra.main(config_path=".", config_name="config_face_vit", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    if opt.wandb.enabled:
        wandb.init(config=opt, project=opt.wandb.project, group=opt.wandb.group)
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    validate_or_test(opt, model, "val", epoch=opt.training.epochs)

    if opt.training.final_test:
        validate_or_test(opt, model, "test")


if __name__ == "__main__":
    my_main()
