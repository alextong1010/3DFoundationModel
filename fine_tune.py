# utils
import argparse
import yaml
import os
from tqdm.auto import tqdm
import numpy as np

# torch
import torch
import torch.nn as nn
from torchvision.io import read_image

# hugging face
from transformers import AutoProcessor, CLIPVisionModel
from transformers import get_scheduler


def main():
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/default.yml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    dataset_path = config["dataset_path_dir"]
    model_save_dir = config["model_save_dir"]
    pretrained_weight = config["pretrained_weight"]
    num_epochs = config["num_epochs"]
    eval_epoch = config["eval_epoch"]
    lr = config["lr"]
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    eps = config["eps"]


    ############### Network ###############
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    nets = nn.ModuleDict({})

    # load foundation model
    nets["foundation_model"] = CLIPVisionModel.from_pretrained(pretrained_weight)

    # transfer model to device
    nets = nets.to(device)

    ############### Dataset ###############
    
    # load image processor
    processor = AutoProcessor.from_pretrained(pretrained_weight)

    # TODO: create dataloader given the dataset path "dataset_path"
    dataloader = 


    ############### Fine-tuning ###############
    # Follow the tutorial in huggingface: https://huggingface.co/docs/transformers/en/training
    # Check the section "Fine-tune a pretrained model in native PyTorch"
    # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset
    optimizer = torch.optim.AdamW(params=nets.parameters(), lr=lr, 
                                 betas=(beta1, beta2), eps=eps)
    
    # Cosine LR schedule with linear warmup
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=num_training_steps
    )

    # set it to training mode (network weights could be fine-tuned)
    nets["foundation_model"].train()

    with tqdm(range(1, num_epochs+1), desc='Epoch') as tglobal:
         # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = []
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    # TODO: compute loss function according to Yichuan's slides
                    loss_2D = 
                    loss_3D = 

                    loss = loss_2D + loss_3D
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # save model
            if (epoch_idx % eval_epoch == 0) or (epoch_idx in [1, num_epochs]):
                checkpoint_dir = '{}/checkpoint_epoch_{}'.format(model_save_dir, epoch_idx)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                for model_name, model in nets.items():
                    model_path = os.path.join(checkpoint_dir, f"{model_name}.pth")
                    torch.save(model.state_dict(), model_path)
                    print(f"{model_name}.pth saved")
                print("All models have been saved successfully.")

                # TODO: evaluate the fine-tuned model
                evaluate(checkpoint_dir)
    



if __name__ == "__main__":
    main()