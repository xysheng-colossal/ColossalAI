import torch
from calflops import calculate_flops
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from colossalai.accelerator import get_accelerator

MODEL_CONFIGS = {
    "7b": Qwen2Config(
        hidden_size=3584,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        max_position_embeddings=131072,
    ),
}

batch_size = 1
max_length = 4096
vocab_size = MODEL_CONFIGS["7b"].vocab_size
input_ids = torch.randint(0, vocab_size, (batch_size, max_length), device=get_accelerator().get_current_device())
attention_mask = torch.ones_like(input_ids)
idx = 0
model_inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "labels": input_ids,
}
model = AutoModelForCausalLM.from_config(
    MODEL_CONFIGS["7b"],
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
flops, macs, params = calculate_flops(model=model, kwargs=model_inputs)

print("Qwen2 FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
