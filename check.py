# pip install bitsandbytes accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# to use 4bit use `load_in_4bit=True` instead
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
checkpoint = "/home/mhaque4/Documents/fineTune/finetuneModels/starcoder2-3b/non-quant/checkpoint-1000"

# model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config)
model = AutoModelForCausalLM.from_pretrained(checkpoint)

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


print(f"Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB")
inputs = tokenizer.encode("def truncate_number(number: float) -> float:", return_tensors="pt").to("cuda")
outputs = model.generate(inputs, min_length=50, max_length=1000)
print(tokenizer.decode(outputs[0]))
