# 3DFoundationModel

## Evaluation Metric of Manipulation Task

### TODOs
- Read and try to understand most of the code in `diffusion_policy/eval_baseline.py` and `diffusion_policy/train_baseline.py`.

- Install the virtual environment by following the **Environment** section from "diffusion_policy/README.md", make sure you install `PyTorch` with the correct cuda version on your device.

- Download example push-T from the [share link](https://drive.google.com/file/d/1fCxkzbv7q7mzsccrTOm3sPCpHEouU5oc/view?usp=sharing) and unzip it to the path: `diffusion_policy/push_t_blue_dataset/domain6.zarr`.

- Register and login `wandb` with your own account.

- Try a run on the existing code by following the **Train** section from "diffusion_policy/README.md". You may want to specify a config path (you can use `diffusionpolicy/configs/baseline.yml` by default) and adjust the value `num_tests` and `eval_epoch` for saving evalutation time. 

- In order to evaluate the manipulation performance given a fine-tuned model, we would like to set `use_pretrained=True` in the config file, so that the training code would freeze the weights of vision encoder.

- Modify evaluation metric in `diffusion_policy/eval_baseline.py`: instead of evaluating a full trajectory given the initial observation image, for each step, we would like to use the groundtruth observation image to predict an action `action_predicted` and do a cosine similarity with `action_groundtruth`. If the length of two actions are not the same, we might want to add a euclidean distance to the metric.

- Design training pipeline #1: for the given dataset, we use the first 90% of demos to train the diffusion model `ConditionalUnet1D`, and then use the rest 10% of demos to test the performance.

- Design training pipeline #2: for the given dataset, we use the first 90% of steps in all demos to train the diffusion model `ConditionalUnet1D`, and then use the rest 10% of steps in all demos to test the performance.

