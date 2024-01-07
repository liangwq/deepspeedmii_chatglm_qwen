# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any

from ...config_v2 import RaggedInferenceEngineConfig
from ..inference_policy_base import ContainerMap, InferenceV2Policy
from .container import ChatGLMNonTransformerContainer,ChatGLMTransformerContainer
from .model import ChatGLMInferenceModel


class ChatGLMPolicy(InferenceV2Policy):

    def instantiate_model(self, engine_config: RaggedInferenceEngineConfig, mp_group: Any) -> ChatGLMInferenceModel:
        return ChatGLMInferenceModel(config=self._model_config, engine_config=engine_config, base_mp_group=mp_group)

    def build_container_map(self) -> ContainerMap:
        map = ContainerMap()

        trans_container_cls = ChatGLMTransformerContainer
        transformer_containers = [trans_container_cls(self.model) for _ in range(self.model.num_layers)]

        map.set_transformer_params(['transformer.encoder.layers'], transformer_containers)

        map.set_non_transformer_params(ChatGLMNonTransformerContainer(self.model))

        #map.set_unmapped_params(['lm_head.weight'])
        map.set_unmapped_params(['transformer.rotary_pos_emb.inv_freq'])
            #[f'model.layers.{i}.self_attn.rotary_emb' for i in range(self.model.num_layers)])
        print(map)
        

        return map