from transformer_heads import create_headed_qlora, load_lora_with_heads
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
)
from transformer_heads.util.helpers import DataCollatorWithPadding, get_model_params
from peft import LoraConfig
from transformer_heads.config import HeadConfig
from transformer_heads.util.model import print_trainable_parameters
from transformer_heads.util.evaluate import (
    evaluate_head_wise,
)
from transformer_heads import load_headed
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
import audioop
import os
import sys

# Need to make sure we are JUST using on GPU
#if int(os.environ['CUDA_VISIBLE_DEVICES']) != 0:
#    print('CAN ONLY TRAIN ON 1 GPU')
#    sys.exit(-1)

def setup_device():
    # Initialize the process group
    torch.distributed.init_process_group(backend='nccl')

    # Set the device for this process
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    return local_rank, device

local_rank, device = setup_device()

model_path = "meta-llama/Llama-2-7b-hf"
#train_batch_size = 2
#eval_batch_size = 2
train_epochs = 1
eval_epochs = 1
logging_steps = 100

model_params = get_model_params(model_path)
model_class = model_params["model_class"]
hidden_size = model_params["hidden_size"]
vocab_size = model_params["vocab_size"]

head_configs = [
    HeadConfig(
        name=f"reduced_output",
        layer_hook=-1,
        in_size=hidden_size,
        output_activation="linear",
        pred_for_sequence=True,
        loss_fct="cross_entropy",
        num_outputs=128,
    ),
    HeadConfig(
        name=f"lm_head",
        layer_hook=-1,
        in_size=hidden_size,
        hidden_size=0,
        num_layers=1,
        output_activation="linear",
        is_causal_lm=True,
        loss_fct="cross_entropy",
        num_outputs=vocab_size,
        is_regression=False,
        output_bias=False,
        trainable=False,
    )
]

raw_data = np.fromfile(f'data/new_data/w000.data', dtype = np.int16)
for i in range(1, 100):
        raw_data = np.append(raw_data, np.fromfile(f'data/new_data/w{i:03}.data', np.int16))

data_iq = raw_data[0::2] + 1j*raw_data[1::2]
signal_real = np.real(data_iq).copy().astype(np.int16) 
signal_real = signal_real.tobytes()
biased = audioop.lin2lin(signal_real, 2, 1)
data = audioop.bias(biased, 1, 2**7)
ndata = np.frombuffer(data, dtype = np.uint8)
data = np.right_shift(ndata, 1)
data_split = sliding_window_view(data, 257)
d = data_split[np.random.choice(data_split.shape[0], 300000, replace=False), :]

inputs = d[:, :256] 
targets = d[:, 256:]

targets = targets.flatten()  # Now targets has shape (x,)

# One-hot encode the targets
one_hot_targets = np.array([val for val in targets])
ascii_inputs = np.array([[''.join(map(chr, row)) for row in inputs]])

# Create a Hugging Face dataset
data = {
    'input': list(ascii_inputs.flatten()),  # Convert to list of lists for compatibility
    'lm_head' : list(ascii_inputs.flatten()),
    'reduced_output': targets #one_hot_targets
}
hf_dataset = Dataset.from_dict(data)
train = hf_dataset

raw_data = np.fromfile(f'data/new_test/t00.data', dtype = np.int16)

data_iq = raw_data[0::2] + 1j*raw_data[1::2]
signal_real = np.real(data_iq).copy().astype(np.int16) 
signal_real = signal_real.tobytes()
biased = audioop.lin2lin(signal_real, 2, 1)
data = audioop.bias(biased, 1, 2**7)
ndata = np.frombuffer(data, dtype = np.uint8)
data = np.right_shift(ndata, 1)
data_split = sliding_window_view(data, 257)
d = data_split[np.random.choice(data_split.shape[0], 10000, replace=False), :]

inputs = d[:, :256] 
targets = d[:, 256:]

targets = targets.flatten()  # Now targets has shape (x,)

# One-hot encode the targets
one_hot_targets = np.array(val for val in targets)
ascii_inputs = np.array([[''.join(map(chr, row)) for row in inputs]])

# Create a Hugging Face dataset
data = {
    'input': list(ascii_inputs.flatten()),  # Convert to list of lists for compatibility
    'lm_head' : list(ascii_inputs.flatten()),
    'reduced_output': targets #one_hot_targets
}
hf_dataset = Dataset.from_dict(data)
test = hf_dataset

dd = DatasetDict({
    'train': train,
    'test': test,
})

tokenizer = LlamaTokenizer.from_pretrained(model_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


def processing_function(examples):
    out = tokenizer(examples['input'], padding=False, truncation=True)
    #out = tokenizer(examples['input'], padding=True, truncation=True)
    out['lm_head'] = out["input_ids"].copy()
    return out

for split in dd.keys():
    dd[split] = dd[split].shuffle()
    dd[split] = dd[split].map(processing_function, batched=True)

dd.set_format(
    type="torch",
    #device = 'cuda',
    columns=["input_ids", "attention_mask"] + [x.name for x in head_configs],
)
for split in dd.keys():
    dd[split] = dd[split].remove_columns(["input"])

quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float32,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = load_headed(
    base_model_class=model_class,
    model_name=model_path,
    quantization_config=quantization_config,
    head_configs=head_configs,
    #device_map='auto'#{"": torch.cuda.current_device()},
    device_map={"": torch.cuda.current_device()},
)

collator = DataCollatorWithPadding(
    feature_name_to_padding_value={
        "input_ids": tokenizer.pad_token_id,
        "attention_mask": 0,
        "lm_head": -100,
        #**{key.name: -100 for key in head_configs},
    }
)

args = TrainingArguments(
    output_dir='test_with_lm_head',
    fp16 = True,
    #auto_find_batch_size = True,
    remove_unused_columns = False,
    per_device_eval_batch_size=8,
    do_eval = False,
    num_train_epochs=1,
    logging_dir='./logs',
    logging_steps=10,
    dataloader_num_workers=4,
    # Enable DDP
    local_rank=local_rank,  # Pass the local rank for DDP
    save_steps = 500,
)

trainer = Trainer(
    model,
    args=args,
    train_dataset=dd["train"],
    data_collator=collator,
)
trainer.train()

if local_rank == 0:
    #model.save_pretrained("test_model_multigpu_bigtrain")
    print('saving')
    model.save_pretrained("test_model_with_lm_head")

print('done')