from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel, GPTNeoXForCausalLM


@dataclass
class HeadOutput:
    logits: torch.FloatTensor


class TransformerWithHead(PreTrainedModel):
    """
    This class initializes the linear head to zeros
    """

    def __init__(self, name, linear_probe=False, reinit_head=True, **kwargs):
        print(kwargs)
        config = AutoConfig.from_pretrained(name, **kwargs)
        super().__init__(config)
        self.num_labels = config.num_labels
        lm = AutoModelForCausalLM.from_pretrained(name, **kwargs)

        self.input_device = None

        if isinstance(lm, GPTNeoXForCausalLM):
            self.lm = lm
            self.transformer = lm.gpt_neox
            self.input_device = self.transformer.embed_in.weight.device
            hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))

        else:
            self.lm = lm
            self.input_device = self.transformer.wte.weight.device
            self.transformer = lm.transformer
        
        hidden_size = getattr(config, "n_embd", getattr(config, "hidden_size", None))

        if reinit_head:
            self.score = torch.nn.Linear(hidden_size, self.num_labels, bias=False).to(
                lm.lm_head.weight.dtype if not isinstance(lm, GPTNeoXForCausalLM) else lm.embed_out.weight.dtype
            )
            torch.nn.init.normal_(self.score.weight, std=0.0)
        else:
            self.score = lm.lm_head if not isinstance(lm, GPTNeoXForCausalLM) else lm.embed_out
        self.linear_probe = linear_probe

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls(name, **kwargs)

    def gradient_checkpointing_enable(self):
        model = self.transformer
        (
            model if hasattr(model, "save_pretrained") else model.module
        ).gradient_checkpointing_enable()

    def forward(self, input_ids: torch.LongTensor):
        """
        Forward pass of the model with a linear head.

        Parameters:
        input_ids (torch.LongTensor): Input tensor containing the token ids.

        Returns:
        HeadOutput: Output dataclass containing the logits.
        """
        input_lens = (input_ids != 0).sum(dim=-1)
        input_ids = input_ids.to(self.input_device)
        transformer_outputs = self.transformer(input_ids)
        hidden_states = torch.stack(
            [transformer_outputs[0][i, input_lens[i] - 1, :] for i in range(len(input_lens))]
        )

        for i in range(hidden_states.shape[0]):
            if torch.any(torch.isnan(hidden_states[i])):
                print("nan hidden states", i)
                print(hidden_states[i])
                print(hidden_states[i].shape)
                print(torch.isnan(hidden_states[i]).sum())
                raise ValueError

        self.score.to(hidden_states.device)
        if self.linear_probe:
            hidden_states = hidden_states.detach()
        logits = self.score(hidden_states)
        return logits
