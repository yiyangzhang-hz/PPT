from typing import Optional

import torch
from peft import get_peft_model
from transformers import BertModel

from model.inputer.llm_concat_inputer import BertConcatInputer
from model.operators.base_llm_operator import BaseLLMOperator


class BertOperator(BaseLLMOperator):
    inputer_class = BertConcatInputer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.transformer = BertModel.from_pretrained(self.config.llm_dir)  # type: BertModel
        self.transformer.embeddings.word_embeddings = None
        self.layer_split(self.transformer.config.num_hidden_layers)

    def _slice_transformer_layers(self):
        self.transformer.encoder.layer = self.transformer.encoder.layer[self.config.layer_split + 1:]

    def _lora_encoder(self, peft_config):
        self.transformer.encoder = get_peft_model(self.transformer.encoder, peft_config)
        self.transformer.encoder.print_trainable_parameters()

    def get_all_hidden_states(
            self,
            hidden_states,
            attention_mask,
    ):
        bert = self.transformer

        input_shape = hidden_states.size()[:-1]
        device = hidden_states.device

        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        hidden_states = bert.embeddings(
            token_type_ids=token_type_ids,
            inputs_embeds=hidden_states,
        ) #在原先词嵌入的基础上，添加位置嵌入和类型嵌入后得到的综合嵌入结果，相当于还多了一点上下文信息

        return self._layer_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

    def _layer_forward(
        self,
        hidden_states: torch.FloatTensor, #存在某一层的每篇新闻的tokens序列的embedding序列
        attention_mask: torch.Tensor,
    ):
        bert = self.transformer

        attention_mask = bert.get_extended_attention_mask(attention_mask, hidden_states.size()[:-1])
        all_hidden_states = ()

        for i, layer_module in enumerate(bert.encoder.layer):
            all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )
            hidden_states = layer_outputs[0]

        all_hidden_states = all_hidden_states + (hidden_states,) #每一层的512*30*768的hidden_states，BERT一共有12层，包括输入输出的话一共有13组数据

        return all_hidden_states

    def layer_forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.Tensor,
    ):
        return self._layer_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )[-1]
