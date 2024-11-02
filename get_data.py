"""Main data fetcher"""

from typing import Generator, Callable
import numpy as np
import audioop
import logging.config
import os
import sys
from numpy.lib.stride_tricks import sliding_window_view
import constants
import random

import matplotlib.pyplot as plt

def our_norm(x):
    maxx = max(x)
    minx = min(x)
    old_range = maxx - minx
    new_range = 127
    nx = [(((o - minx) * new_range) // old_range) for o in x]
    return bytes(nx)

def fetch_preprocessed(
            which: str, 
            ) -> Generator [any, any, any]:

    # Logging config
    logging.config.dictConfig(constants.LOGGING_CONFIG)
    logger = logging.getLogger(__name__) 
    
    logger.debug(f'Fetching preprocessed data {which}')

    data_list = []
    
    for i in range(1000):
        logger.debug(f'Fetching {i}th data')
        with np.load(f'data/chunked_data/new_c256/d{i}.data.npz', allow_pickle=True) as f:
            data_list.append(f['dat'].astype(np.uint8))

    d = np.concatenate(data_list, axis=0)
    return map(lambda x: x.tobytes(), d.astype(np.uint8))


def fetch(stream_mode : bool,
          amt : int, # number of chunks for non stream and length of stream for stream
          context_window: int, # needed for non stream
          filename: int, 
          scale: float = 1,
          offset: int = 0,
          map_fn: Callable = None,
          rchunks : bool = False,
          noise : float = None, # None for no noise otherwise the standard deviation of noise
          random_order: bool = False,
          ) -> Generator[any, any, any]:
    """Returns chunks of float data which are fetched from wifi data

    Args:
        n_chunks: Number of chunks to fetch. If -1 works in stream mode and assumes test data
        context_window: How large each context window is.
        filename: Integer describing which file to fetch from, if -1, will ignore and start from 0 and go until n_chunks limit is reached.
        normalisation: Default True, will transform the data so the maximum value is 255 and the minimum value is 0.
    """

    # Logging config
    logging.config.dictConfig(constants.LOGGING_CONFIG)
    logger = logging.getLogger(__name__) 

    # All file names should be in the format data/wX.data where X is an integer.
    if stream_mode:
        logger.debug(f'Fetching data in stream mode with stream length {amt}')
        context_window = amt
        n_chunks = 1
        
    else:
        # TODO this is wrong if we do all
        logger.debug(f'Fetching data in chunk mode with {"all" if filename == -1 else amt} chunks each size {context_window}')
        n_chunks = amt

    # Load in data 
    if filename == -99: # All test files
        n_chunks = amt
        logger.debug('Loading all test data')
        n_files = len(next(os.walk('data/new_test'))[2])
        data = np.fromfile('data/new_test/t00.data', np.int16)
        for i in range(1, n_files - 1):
            data = np.append(data, np.fromfile(f'data/new_test/t{i:02}.data', np.int16))

    elif filename == -1: 
        n_chunks = 2e11
        logger.debug('Loading all data')
        #n_files = len(next(os.walk('data/signal_final'))[2])
        n_files = 1000
        data = np.fromfile('data/signal_final/w0000.data', np.int16)
        all_data = np.zeros((n_files, len(data)), dtype = np.int16)
        all_data[0] = data
        print(n_files)
        for i in range(1, n_files):
            all_data[i] = np.fromfile(f'data/signal_final/w{i:04}.data', np.int16)

        data = all_data.flatten()
        #data = np.fromfile('data/ieee_data/d0.data', np.int16)
        #data = np.append(data, np.fromfile('data/ieee_data/d1.data', np.int16))
    elif filename <=-2: # Test file is -2
        logger.debug(f'Loading {(-filename - 2):02}th test file with offset {offset}')
        data = np.fromfile(f'data/new_test/t{(- filename - 2):02}.data', np.int16)
        # TEMP
        #data = np.fromfile(f'data/ieee_data/d2.data', np.int16)
        #current_min = data.min()
        #current_max = data.max()

        #new_min = np.iinfo(np.int16).min  # -32768
        #new_max = np.iinfo(np.int16).max  # 32767

        #scaled_data = ((data - current_min) / (current_max - current_min)) * (new_max - new_min) + new_min
        #data = scaled_data.astype(np.int16)

        logger.debug(f'Ofsetting by {offset}')
        data = data[offset:]
    else:
        logger.debug(f'Loading from file {filename:04} with offset {offset}')
        data = np.fromfile('data/signal_final/w{filename:04}.data', np.int16)
        data = data[offset:]



    data_iq = data[0::2] + 1j*data[1::2]
    if scale != 1.0:
        logger.debug(f'Scaling by factor {scale}')

    signal_real = (scale*np.real(data_iq).copy()).astype(np.int16) # Add a scaling factor?

    if noise is not None:
        signal_real = signal_real + np.random.normal(0,noise,signal_real.shape).astype(np.int16)

    signal_real = signal_real.tobytes()

    new_signal = audioop.lin2lin(signal_real, 2, 1)
    new_signal = audioop.bias(new_signal, 1, 2**7)

    def _extract_rf_slide(sample: bytes):
        x = np.frombuffer(sample, dtype = np.uint8)
        if rchunks:
            logger.info(f'rchunks true, doing array_split')
            patches = np.array_split(x,
                range(
                    context_window,
                    len(sample),
                    context_window,))
        else:
            logger.info("sliding window data")
            patches = sliding_window_view(x, context_window).copy()

        if len(patches[-1]) != context_window:
            patches.pop()

        if random_order:
            logger.info(f'Randomly shuffling split array')
            patches = np.random.permutation(patches)

        if map_fn is not None:
            return map(map_fn, patches)

        #return map(lambda patch: our_norm(patch), patches)
        return map(lambda patch: patch.tobytes(), patches)

    idx = 0
    for patch in _extract_rf_slide(new_signal):
        if idx == n_chunks:
            return
        yield patch
        idx += 1
