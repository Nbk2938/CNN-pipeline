#!/bin/bash
# env_setup.sh  –  ONE-TIME installer (5–10 min)

module purge
module load stack/2024-06  gcc/12.2.0  cuda/12.1.1  python/3.10.13  eth_proxy

python3.10 -m ensurepip --upgrade --default-pip #HOTFIX :D

# remove anything left from previous attempts
pip uninstall -y torch torchvision numpy "nvidia-*" triton || true

pip install --user "numpy<2" \
  torch==2.2.1+cu121  torchvision==0.17.1+cu121 \
  matplotlib  pytorch_msssim \
  --extra-index-url https://download.pytorch.org/whl/cu121

