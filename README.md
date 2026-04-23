# scai-jam-nautilus
Kubernetes scripts for the SCAI Jam #1 in Spring 2026

Welcome to the SCAI Jam MLOps Starter Guide! This repository contains the essential templates required to train machine learning models using Nautilus Kubernetes, Hugging Face, and Weights & Biases (W&B).

## Prerequisites
1. **GitHub Repository**: Create a repository for your team. You can use the `train_template.py` provided here as a starting point (`train.py`) in your repository.
2. **Weights & Biases Account**: Get your API key from [wandb.ai/authorize](https://wandb.ai/authorize).
3. **Hugging Face Account**: Get an Access Token with `write` permissions from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

## Step-by-Step Guide

### Step 1: Configure Your Secrets
We use a Kubernetes Secret to safely pass tokens to your training job without putting them in code.

1. Open `secrets.yaml`.
2. Replace `your_wandb_api_key_here` and `your_huggingface_write_token_here` with your actual keys.
3. Keep `namespace: scai-jam` unless instructed otherwise.
4. Apply the secrets to the cluster:
   ```bash
   kubectl apply -f secrets.yaml
   ```

### Step 2: Set up Your Training Script
You need a Python script to actually do the training. Take a look at `train_template.py`.
1. Upload this file as `train.py` to your team's GitHub repository.
2. Modify the model, dataset, hyperparameters or anything else as you see fit!

### Step 3: Configure the Training Job
The `train.yaml` file defines the environment, resources (GPUs/RAM), and the commands needed to kick off the training.

1. Open `train.yaml`.
2. Find `<GROUP_NAME>` and replace it with your team's name.
3. In the `args` section of the container, replace `<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>` with your newly created GitHub repository.
4. Save the file and submit the job to the cluster:
   ```bash
   kubectl apply -f train.yaml
   ```

### Step 4: Monitor your Job
Once submitted, you can check on your job and view the logs in real-time.

1. View all running pods (find the one with your group name):
   ```bash
   kubectl get pods -n scai-jam
   ```
2. View the logs (replace `POD_NAME` with the name from the previous step):
   ```bash
   kubectl logs -f POD_NAME -n scai-jam
   ```
3. Head over to [Weights & Biases](https://wandb.ai/) to see your live training metrics (loss, accuracy, etc.)!
4. Check [Hugging Face](https://huggingface.co/) after training finishes to see your pushed model (if enabled).

### Helpful K8s Commands
- To stop/delete your job: `kubectl delete -f train.yaml`
- To get into an interactive shell inside a running pod (debugging): `kubectl exec -it POD_NAME -n scai-jam -- /bin/bash`

