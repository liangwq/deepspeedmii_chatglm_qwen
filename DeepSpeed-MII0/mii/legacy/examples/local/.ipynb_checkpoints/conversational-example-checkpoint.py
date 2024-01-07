# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
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
