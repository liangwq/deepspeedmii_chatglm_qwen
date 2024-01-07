# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
'''
import mii

mii_configs = {'tensor_parallel': 1}

# gpt2
name = "facebook/opt-1.3b" #microsoft/DialoGPT-large"
model_config = {
        "model_path": "/root/.cache/huggingface/hub/models--facebook--opt-1.3b",
        "enable_deepspeed": True,
        "enable_zero": True,
    }

print(f"Deploying {name}...")

mii.deploy(task='conversational', model=name,model_config = model_config , deployment_name=name + "_deployment")



import mii
from transformers import AutoConfig

# put your model name here
MODEL_NAME="facebook/opt-1.3bB"
MODEL_DEPLOYMENT_NAME=MODEL_NAME+"_deployment"


mii_config = {"dtype": "fp16"}

name = MODEL_NAME

config = AutoConfig.from_pretrained(name)


ds_config = {
    "fp16": {
        "enabled": True
    },
    "bf16": {
        "enabled": False
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory":True
        }
    },
    "train_batch_size": 1,
}

mii.deploy(task='text-generation',
           model=name,
           deployment_name=MODEL_DEPLOYMENT_NAME,
           model_path="/root/.cache/huggingface/hub/" + name,
           mii_config=mii_config,
           ds_config=ds_config,
           enable_deepspeed=False,
           enable_zero=True)
'''
# DeepSpeed Team
import mii

mii_configs = {'tensor_parallel': 1}

# gpt2
name = "microsoft/DialoGPT-large"

print(f"Deploying {name}...")

mii.deploy(task='conversational', model=name, deployment_name=name + "_deployment")