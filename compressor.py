# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Runs the compressor"""

from collections.abc import Generator
import functools
import time
from typing import Callable, Mapping

import math
import tqdm
import numpy as np
import gzip
import random
import utils

import language_model
import constants
import logging.config

import llama.compress

COMPRESS_FN_DICT = { #Mapping[str, Compressor] = {
    'gzip': functools.partial(gzip.compress, compresslevel=9),
    'btransformer': language_model.compress,
    'btransformer_smooth': language_model.compress_smooth,
    'llama': llama.compress.fo_compress,
}
#def evaluate_smooth_compressor(
#    compress_fn_name: str,
#    data: Generator,
#    mask_fn : Callable,
#    n_chunks: int, # These should be moved to a global config
#    context_l: int,
#    use_tqdm: bool = True,
#) -> tuple[float, float]:
#  """Evaluates the compressor on the chunked dataset.
#
#  Args:
#    compress_fn: The function that evaluates data.
#    data: List of the data to consider
#    mask_fn: The function that masks the data in case the compressor cannot
#      handle all possible byte values (e.g., language models can only process
#      ASCII-decodable data).
#    n_chunks: Should be set in a global context 
#    context_l: Context window
#    use_tqdm: Whether to use a progress bar or not.
#
#  Returns:
#    The compression rate and the total running time.
#  """
#
#  # Logger
#  logging.config.dictConfig(constants.LOGGING_CONFIG)
#  logger = logging.getLogger(__name__) 
#
#  logger.info(f'Smoothly compressing data of length {context_l * n_chunks=} with {compress_fn_name}')
#
#  num_missed_bits = running_time = raw_length = compressed_length = 0
#
#  compress_fn = COMPRESS_FN_DICT[compress_fn_name]
#
#  logger.info(f'Compressing full batch')
#  if mask_fn is not None:
#    data, missed_bits = mask_fn(data)
#    num_missed_bits += missed_bits
#
#  t0 = time.perf_counter()
#  compressed_data = compress_fn(data, compress_fn_name)
#  t1 = time.perf_counter()
#
#  running_time += t1 - t0
#  #raw_length += len(data-context_l)
#  raw_length += len(data)
#  compressed_length += len(compressed_data)
#
#  # Since language models are trained on ASCII strings, they cannot handle all
#  # byte values. Thus, we mask the data to be ASCII-decodable by zeroing
#  # `num_missed_bits` of the most significant bits. However, this means that we
#  # are effectively only compressing `num_bits - num_missed_bits` bits, so we
#  # rescale the `compressed_length` to account for this.
#  if mask_fn is not None:
#    num_bits = 8 * n_chunks * context_l
#    compressed_length *= num_bits / (num_bits - num_missed_bits)
#
#  # We only count the header once for classical compressors.
#  if compress_fn_name == 'gzip':
#    header_length = len(compress_fn((0).to_bytes(1, 'little')))
#    compressed_length -= header_length * (n_chunks - 1) 
#
#  return compressed_length / raw_length, running_time
#
def evaluate_compressor(
    compress_fn_name: str,
    params: str,
    config,
    data: Generator,
    mask_fn : Callable,
    sigma : float = 3.0,
    n_chunks: int = 1, #  this is fixed
    context_l: int = 1,
    use_tqdm: bool = True,
) -> tuple[float, float]:
  """Evaluates the compressor on the chunked dataset.

  Args:
    compress_fn: The function that evaluates data.
    data: List of the data to consider
    mask_fn: The function that masks the data in case the compressor cannot
      handle all possible byte values (e.g., language models can only process
      ASCII-decodable data).
    n_chunks: Should be set in a global context 
    context_l: Context window
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The compression rate and the total running time.
  """

  # Logger
  logging.config.dictConfig(constants.LOGGING_CONFIG)
  logger = logging.getLogger(__name__) 

  num_missed_bits = running_time = raw_length = compressed_length = 0
  compress_fn = COMPRESS_FN_DICT[compress_fn_name]

  len_data = 0
  len_datum = 0
  kld_sum = 0

  #x_ld = 0
  #x_cl = 0

  r_list = []
  for i, datum in enumerate(data):
    # Get stats for context_l and n_chunks

    len_data += 1
    len_datum = len(datum)

    logger.debug(f'Compressing {i}th batch')
    if mask_fn is not None:
      datum, missed_bits = mask_fn(datum)
      num_missed_bits += missed_bits

    # Preprocess data (see this still works)
    #logger.info('Preprocessing')
    #datum = utils.our_norm(datum)

    t0 = time.perf_counter()
    if compress_fn_name == 'gzip':
      compressed_data = compress_fn(datum)
      pad = 0
    else:
      compressed_data, pad, kld = compress_fn(datum, params, config, sigma)
      #x_ld += 8
      #x_cl += (len(compressed_data) * 8) - pad + 1
      kld_sum += kld
    t1 = time.perf_counter()
    logger.debug(f'Returned {len(compressed_data)} bytes')
    logger.debug(f'Rate was {(len(compressed_data) + len(datum)/8)/len(datum)}')
    logger.debug(f'Required {pad} padding bits')

    running_time += t1 - t0
    raw_length += len(datum)
    compressed_length += len(compressed_data)
    r_list.append((len(compressed_data)+32)/(256))
    logger.debug(f'last appended to rlist was {r_list[-1]=}')

  fc = len_datum * len_data
  kld = kld_sum/(i+1)
  logger.info(f'Average KLD was {kld}')
  # Since language models are trained on ASCII strings, they cannot handle all
  # byte values. Thus, we mask the data to be ASCII-decodable by zeroing
  # `num_missed_bits` of the most significant bits. However, this means that we
  # are effectively only compressing `num_bits - num_missed_bits` bits, so we
  # rescale the `compressed_length` to account for this.

  # If we use the smooth compressor we need to subtract 
  import yaml
  with open(f'params/{params}/info.yml', 'r') as f:
    info = yaml.safe_load(f)

  # Only if we offset
  #if compress_fn_name == 'btransformer_smooth':
  #  raw_length -= info['cw']
  #  fc -= info['cw']

  if mask_fn is not None:
    # Add one bit per missed bit, round up to nearest 8
    compressed_length += int(math.ceil(fc / 8.0)) 


  # We only count the header once for classical compressors.
  if compress_fn_name == 'gzip':
    header_length = len(compress_fn((0).to_bytes(1, 'little')))
    compressed_length -= header_length * (len_data - 1) 

  return compressed_length / raw_length, running_time, r_list

"""
def main(_) -> None:
  logging.info('Compressor: %s', _COMPRESSOR.value)
  logging.info('Dataset: %s', _DATASET.value)

  compress_fn = compressor.COMPRESS_FN_DICT[]
  get_data_generator_fn = functools.partial(
      data_loaders.GET_DATA_GENERATOR_FN_DICT[_DATASET.value],
      num_chunks=_NUM_CHUNKS.value,
  )

  if _COMPRESSOR.value in compressor.COMPRESSOR_TYPES['classical']:
    unchunked_rate, unchunked_time = evaluate_compressor_unchunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
    )
    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
        count_header_only_once=True,
        mask_fn=None,
    )
    logging.info(
        'Unchunked: %.1f [%.1fs]', 100 * unchunked_rate, unchunked_time
    )
    logging.info('Chunked: %.1f [%.1fs]', 100 * chunked_rate, chunked_time)

  elif _COMPRESSOR.value in compressor.COMPRESSOR_TYPES['arithmetic_coding']:
    # To compress bytes data, we convert it first to ASCII.
    if _DATASET.value == 'enwik9':
      # For Enwik9, some characters are UTF-8 but not ASCII, so we still need
      # to do the conversion.
      mask_fn = utils.zero_most_significant_bit_if_not_ascii_decodable
    else:
      #mask_fn = utils.right_shift_bytes_by_one
      mask_fn = None

    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
        count_header_only_once=False,
        mask_fn=mask_fn,
    )
    logging.info('Chunked: %.1f [%.1fs]', 100 * chunked_rate, chunked_time)


if __name__ == '__main__':
  app.run(main)
"""