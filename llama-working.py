# Use a pipeline as a high-level helper
from transformers import pipeline
from transformers import (
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch

torch_dtype = torch.float16
attn_implementation = "eager"
base_model = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.add_special_tokens({"pad_token":"<pad>"})

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)

messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe(messages))