import time
from transformer_heads import create_headed_qlora, load_lora_with_heads
import gc
import pickle
import utils
import arithmetic_coder
from torch.nn import functional as F
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    Trainer,
    BitsAndBytesConfig,
    TrainingArguments,
    GenerationConfig,
)
import constants
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

from transformer_heads.util.evaluate import (
    evaluate_head_wise,
    get_top_n_preds,
    get_some_preds,
)
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

import random

model_path = "meta-llama/Llama-2-7b-hf"
model_params = get_model_params(model_path)
model_class = model_params["model_class"]
hidden_size = model_params["hidden_size"]
vocab_size = model_params["vocab_size"]

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
    model_class,
    model_path,
    head_folder_path="test_with_lm_head/checkpoint-9375",
    device_map="cuda",
    quantization_config = quantization_config,
)

import logging
logging.config.dictConfig(constants.LOGGING_CONFIG)
logger = logging.getLogger()

pdf_llama = pd.DataFrame(0, index = np.arange(10), columns = constants.SNR_SET, dtype = np.float64)
pdf_reduced_head = pd.DataFrame(0, index = np.arange(10), columns = constants.SNR_SET, dtype = np.float64)
cw = 1024
runs = 500

#for scale_index in constants.SCALES:
#for scale_index in [0.9, 0.8, 0.7]:
for file_index in range(constants.NUM_TEST_FILES):
    logger.info(f'file: {file_index}')

    for snr_index in constants.SNR_SET:
    #for snr_index in [999]:
        logger.info(f'snr: {snr_index=}')

        raw_data = np.fromfile(f'data/new_test/t{file_index:02}.data', dtype = np.int16)

        data_iq = raw_data[0::2] + 1j*raw_data[1::2]
        signal_real = np.real(data_iq).copy().astype(np.int16) 
        # Add noise here if any
        #signal_real = (scale_index*np.real(data_iq).copy()).astype(np.int16)
        if snr_index != 999:
            signal_real = signal_real + np.random.normal(0,constants.SNR_SD_TABLE.loc[snr_index, 1],signal_real.shape).astype(np.int16)

        signal_real = signal_real.tobytes()
        biased = audioop.lin2lin(signal_real, 2, 1)
        data = audioop.bias(biased, 1, 2**7)
        ndata = np.frombuffer(data, dtype = np.uint8)
        data = np.right_shift(ndata, 1)
        data_split = sliding_window_view(data, cw+1).copy()


        np.random.shuffle(data_split)

        tokenizer = LlamaTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token


        testing_runs = runs
        running_total = 0
        lm_head_pdf_collect = list()
        lm_head_symbol_collect = list()
        reduced_pdf_collect = list()
        reduced_symbol_collect = list()
        raw_len = 0
        start = time.time()
        for idx in range(testing_runs):

            selection = data_split[idx]
            ascii_selection = ''.join(chr(x) for x in selection)


            target = ascii_selection[-1]
            ascii_select_d = ascii_selection[:-1]

            tk = tokenizer.encode(ascii_select_d)
            raw_len += len(tokenizer.decode(tk[-1]))

            batch = tokenizer(ascii_select_d, return_tensors = 'pt')

            with torch.no_grad():
                output = model(**batch)

            torch.cuda.empty_cache()
            gc.collect()

            with torch.no_grad():
                torch.cuda.empty_cache()

            reduced_pdf_collect.append(output['preds_by_head']['reduced_output'][0][-1].clone().detach())
            reduced_symbol_collect.append(target)

            lm_head_pdf_collect.append(output['preds_by_head']['lm_head'][0][-2].clone().detach())
            lm_head_symbol_collect.append(tk[-1])

        enc_output = list()
        encoder = arithmetic_coder.Encoder(
            base=2,
            precision=32,
            output_fn=enc_output.append,
        )

        for symbol, logits in zip(reduced_symbol_collect, reduced_pdf_collect):
            dist = F.softmax(logits).numpy()
            pdf = utils.normalize_pdf_for_arithmetic_coding(dist)
            encoder.encode(pdf, ord(symbol))

        encoder.terminate()
        compressed_bits = ''.join(map(str, enc_output))
        compressed_bytes, padding = utils.bits_to_bytes(compressed_bits)

        clen = len(compressed_bytes) + testing_runs/8

        pdf_reduced_head.loc[file_index, snr_index] = clen/testing_runs

        enc_output = list()
        encoder = arithmetic_coder.Encoder(
            base=2,
            precision=32,
            output_fn=enc_output.append,
        )

        for symbol, logits in zip(lm_head_symbol_collect, lm_head_pdf_collect):
            dist = F.softmax(logits).numpy()
            pdf = utils.normalize_pdf_for_arithmetic_coding(dist)
            encoder.encode(pdf, symbol)

        encoder.terminate()
        end = time.time()
        compressed_bits = ''.join(map(str, enc_output))
        compressed_bytes, padding = utils.bits_to_bytes(compressed_bits)
        print(f'time taken was {end-start}')

        clen = len(compressed_bytes) + raw_len/8
        pdf_llama.loc[file_index, snr_index] = clen/raw_len

        ## save 
        pdf_llama.to_pickle('llama_results/pdf_llama_over_files_final.pkl')
        pdf_reduced_head.to_pickle('llama_results/pdf_reduced_head_over_files_final.pkl')





