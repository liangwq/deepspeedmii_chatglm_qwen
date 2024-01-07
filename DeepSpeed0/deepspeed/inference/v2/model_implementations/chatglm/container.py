# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Create a container object to save model-specific tensors using the policy file above.

from ..common_parameters import *
from ..layer_container_base import LayerContainer
'''
 # HF Chatglm 6b model looks like this:

ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (embedding): Embedding(
      (word_embeddings): Embedding(65024, 4096)
    )
    (rotary_pos_emb): RotaryEmbedding()
    (encoder): GLMTransformer(
      (layers): ModuleList(
        (0-27): 28 x GLMBlock(
          (input_layernorm): RMSNorm()
          (self_attention): SelfAttention(
            (query_key_value): Linear(in_features=4096, out_features=4608, bias=True)
            (core_attention): CoreAttention(
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
            (dense): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (post_attention_layernorm): RMSNorm()
          (mlp): MLP(
            (dense_h_to_4h): Linear(in_features=4096, out_features=27392, bias=False)
            (dense_4h_to_h): Linear(in_features=13696, out_features=4096, bias=False)
          )
        )
      )
      (final_layernorm): RMSNorm()
    )
    (output_layer): Linear(in_features=4096, out_features=65024, bias=False)
  )
)
'''

class ChatGLMTransformerContainer(LayerContainer):
    """
        Transformer layer container for the Chatglm model.
    """
    attn_norm_gamma: NormParameter
    qkv_w: FusedQKVParameter
    qkv_b: FusedQKVParameter
    attn_out_w: AttentionOutputParameter
    attn_norm_beta: NormParameter
    mlp_1_w: MLP1Parameter
    mlp_2_w: MLP2Parameter

    PARAM_MAPPING = {
        "input_layernorm":"attn_norm_gamma.params",
        "self_attention.query_key_value.weight": "qkv_w.params",
        "self_attention.query_key_value.bias": "qkv_b.params",
        "self_attention.dense.weight": "attn_out_w.params",
        "post_attention_layernorm": "attn_norm_beta.params",
        "mlp.dense_h_to_4h.weight": "mlp_1_w.params",
        "mlp.dense_4h_to_h.weight": "mlp_2_w.params",
    }


class ChatGLMNonTransformerContainer(LayerContainer):
    """
        Non-Transformer layer container for the Falcon model.
    """
    word_emb: EmbeddingParameter
    #word_pos_emb: EmbeddingParameter
    #word_pos_inv_emb: EmbeddingParameter 
    final_norm_gamma: NormParameter
    word_unembed: UnembedParameter

    PARAM_MAPPING = {
        "*transformer.embedding.word_embeddings.weight": "word_emb.params",
        #"*transformer.rotary_pos_emb.inv_freq.weight": "word_pos_emb.params",
        "*transformer.encoder.final_layernorm.weight": "final_norm_gamma.params",
        "*transformer.output_layer.weight": "word_unembed.params",
    }

