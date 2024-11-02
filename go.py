"""Main experiment runner"""

import logging.config
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm
import pickle
import xarray as xr
import shutil

import time
import constants
import get_data
import compressor

from btransformer.transformer import TransformerConfig
import btransformer.train
import language_model
import utils
import os
import functools

# Arg-parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--action', dest='action', required=True)
parser.add_argument('--compressor', dest='compressor')
parser.add_argument('--amt', dest='amt')
parser.add_argument('--cw', dest='cw')
parser.add_argument('--which', dest='which')
parser.add_argument('--stream', dest='stream',
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--shh', dest='shh', action=argparse.BooleanOptionalAction)
parser.add_argument('--rchunks', dest='rchunks',
                    action=argparse.BooleanOptionalAction)
parser.add_argument('--file', dest='file')
parser.add_argument('--chunks', dest='chunks')
parser.add_argument('--offset', dest='offset')
parser.add_argument('--scale', dest='scale')
parser.add_argument('--noise', dest='noise')
parser.add_argument('--sigma', dest='sigma')
args = parser.parse_args()

# Logger
logging.config.dictConfig(constants.LOGGING_CONFIG)
logger = logging.getLogger()


def eval():

    # Must do a thorough analysis of compressing ability over the ten test sets
    # Don't forget to do the scaling one but can do that later
    # Default scenario is test over the 10 different training data
    # In this function, amt will be the number of chunks to fetch randomly from each file

    logger.info('--- EVALUATING PARAMETER SET---')

    # Get the parameter set
    which = args.which
    if not os.path.isdir(f'params/{which}'):
        logger.error(f'{which} is not a valid parameter set, quitting.')
        sys.exit(-1)

    logger.info(f'Evaluating parameter set {which}')
    with open(f'params/{which}/info.yml', 'r') as f:
        info = yaml.safe_load(f)

    rope = int(info.get('rope', 0))
    print(f'rope is {rope}')

    config = TransformerConfig(
        vocab_size=info['vocab_size'],
        embedding_dim=info['embedding_dim'],
        num_heads=info['num_heads'],
        num_layers=info['num_layers'],
        emb_init_scale=info['emb_init_scale'],
        widening_factor=info['widening_factor'],
    )
    cw = info['cw']

    # Strictly not stream
    logger.info('Model information dump')
    logger.info(f'{config}')

    # Consider the 10 test files with 10 SNRS from constants file
    #df = pd.DataFrame(
    #    columns=[x for x in constants.SNR_SET],
    #    index=[x for x in range(constants.NUM_TEST_FILES)],
    #)
    amt = int(args.amt)
    da = xr.DataArray(
        np.zeros((len(constants.SNR_SET), len(constants.SCALES), amt)),
        dims = ['snr', 'scale', 'sample'],
        coords = {'snr' : constants.SNR_SET, 'scale': constants.SCALES, 'sample' : np.arange(amt)}
    )

    # Iterate over scale and snr
    for scale in constants.SCALES:
    #for scale in [1.0]:
        logger.info(f'Considering scale {scale}')
        for snr_set in constants.SNR_SET:
        #for snr_set in [30]:
            logger.info(f'Considering SNR {snr_set}')

            # Get data   
            logger.info(f'Noise SD is {constants.SNR_SD_TABLE.loc[snr_set, scale]}')
            noise =constants.SNR_SD_TABLE.loc[snr_set, scale] 
            if snr_set == 999:
                noise = None

            data = get_data.fetch(
                stream_mode=False,
                amt=int(args.amt),
                context_window=cw,
                filename=-99,
                scale=scale,
                offset=0,
                map_fn=None,
                rchunks=False, # This changes everything
                random_order = True,
                noise=noise,
            )
            
            # Mask
            mask_fn = None
            if info['vocab_size'] == 128:
                mask_fn = utils.right_shift_bytes_by_one

            # Evaluate Compressor
            rate, time, r_list = compressor.evaluate_compressor(
                compress_fn_name='btransformer',
                params=which,
                config=config,
                data=data,
                mask_fn=mask_fn,
            )
            da.loc[{'snr' : snr_set, 'scale' : scale}] = r_list
            #print(da.loc[{'snr' : snr_set, 'scale' : scale}])
            #sys.exit(-1)
            # Store value
            da.to_netcdf(f'params/{which}/results.nc')

    # Exit
    return

    

   # # Now we need to loop over test files AND SNR
   # for test_file in range(constants.NUM_TEST_FILES):
   #     logger.info(f'Considering test file {test_file}')

   #     for snr_sd, snr_set in zip(constants.SNR_SD, constants.SNR_SET):
   #         logger.info(f'Considering SNR {snr_set}')

   #         data = get_data.fetch(
   #             stream_mode=False,
   #             amt=int(args.amt),
   #             context_window=cw,
   #             filename=(-2 - test_file),
   #             scale=1,
   #             offset=0,
   #             map_fn=None,
   #             rchunks=True,
   #             random_order = True,
   #             noise=snr_sd,
   #         )

   #         mask_fn = None
   #         if info['vocab_size'] == 128:
   #             mask_fn = utils.right_shift_bytes_by_one

   #         rate, time = compressor.evaluate_compressor(
   #             compress_fn_name='btransformer',
   #             params=which,
   #             config=config,
   #             data=data,
   #             mask_fn=mask_fn,
   #         )

   #         df.loc[test_file, snr_set] = rate



    print(df)

    # Save pickle file
    with open(f'params/{which}/df.pkl', 'wb') as f:
        pickle.dump(df, f)

   # ax = sns.boxplot(df, color='white')
   # plt.setp(ax.artists, edgecolor='k', facecolor='w')
   # plt.setp(ax.lines, color='k')
   # ax.set_xlabel('SNR [dB]')
   # ax.set_ylabel('Compression Rate')
   # ax.set_title(f'{which} Compression Rate against SNR levels')
   # ax.yaxis.set_major_formatter(ticker.PercentFormatter(1, 0))
   # plt.show()

   # plt.savefig(f'params/{which}/results.png')
   # plt.close()


def train():

    # Check which is a valid folder
    logger.info('---- BEGINNING TRAINING ----')
    which = args.which
    if not os.path.isdir(f'params/{which}'):
        logger.error(f'{which} is not a valid parameter set, quitting.')
        sys.exit(-1)

    # If it is a valid folder there are two outcomes, it is new training
    # Or there are parameters already there and we continue training
    with open(f'params/{which}/info.yml', 'r') as f:
        info = yaml.safe_load(f)

    new_train = not os.path.isfile(f'params/{which}/params.npz')

    if new_train:
        logger.info(f'New parameters')
    else:
        logger.info(f'Old parameters with {info["training"]} runs')

    logger.info(
        f'Training with an extra {args.amt} steps with batch size {info["bs"]}')

    config = TransformerConfig(
        vocab_size=info['vocab_size'],
        embedding_dim=info['embedding_dim'],
        num_heads=info['num_heads'],
        num_layers=info['num_layers'],
        emb_init_scale=info['emb_init_scale'],
        widening_factor=info['widening_factor'],
    )

    logger.info(f'Parameters for transformer are {config=}')

    t0 = time.perf_counter()
    #epoch = int(info['epoch'])
    #print(f'{epoch=}')
    params, loss = btransformer.train.train_transformer_decoder(
        new_train=new_train,
        which=which,
        config=config,
        training_steps=int(args.amt),
        cw=info['cw'],
        log_every=int(args.amt)//100,
        batch_size=info['bs'], #* 2**epoch,
        use_tqdm=not args.shh,
        epoch = 0,
    )
    t1 = time.perf_counter()
    running_time = t1 - t0
    logger.info(
        f'{args.amt} training run complete (total {info["training"] + int(args.amt)}) in {running_time} seconds with loss {loss}')

    info['training'] += int(args.amt)
    #info['epoch'] += 1

    np.savez(f'params/{which}/params.npz', **params)
    logger.info(f'Saved params in params/{which}/params.npz file')

    #logger.info(f'Copying params into epoch file params{epoch}.npz')
    #shutil.copy(f'params/{which}/params.npz', f'params/{which}/params{epoch}.npz')

    # Rewrite yaml
    with open(f'params/{which}/info.yml', 'w') as f:
        yaml.dump(info, f, default_flow_style=False)


# TODO this is unsophisticated
def decompress():

    logger.info('Decompressing')
    with open('compressed_bytes.data', 'rb') as f:
        d = f.read()

    logger.info(f'Read in {len(d)} bytes from compressed_bytes.data')

    cw = int(args.cw)
    decompressed_data = language_model.decompress(d, 5, cw)

    p = np.frombuffer(decompressed_data, dtype=np.uint8)
    plt.figure()
    plt.plot(p)
    plt.savefig('figs/tmp/decompressed.png')
    plt.close()


def compress():

    # Options for compressing
    # mRF
    # non-mRF
    # Sequential, cheating
    # Non-sequential, cheating
    # Sequential, non-cheating
    # Non-sequential, non-cheating

    # We want to change the compressor so that it works on a long stream of bytes
    # rather than chunks. If we do end up doing chunking then we can specify that within
    # the compression function

    # What we actually want to do is pass in 'c512_001' and load in the info file
    logger.info('---- BEGINNING COMPRESSION ----')
    compressor_name = args.compressor
    if compressor_name == 'gzip' or compressor_name == 'llama':
        config = None
        cw = int(args.cw)
        info = {}
        info['vocab_size'] = 256
    else:
        with open(f'params/{args.which}/info.yml', 'r') as f:
            info = yaml.safe_load(f)
        config = TransformerConfig(
            vocab_size=info['vocab_size'],
            embedding_dim=info['embedding_dim'],
            num_heads=info['num_heads'],
            num_layers=info['num_layers'],
            emb_init_scale=info['emb_init_scale'],
            widening_factor=info['widening_factor'],
        )
        cw = info['cw']

    sigma = 3.0
    if args.sigma:
        sigma = float(args.sigma)

    # Stream mode
    stream_mode = 0
    if args.stream:
        logging.info('Compressing in stream mode')
        compressor_name += '_smooth'
        stream_mode = 1
        cw = 0

    offset = 0 if not args.offset else int(args.offset)
    scale = 0 if not args.scale else float(args.scale)
    rchunks = 0 if not args.rchunks else 1

    if not args.scale:
        scale = 1.0
    else:
        scale = float(args.scale)

    logging.info(
        f'Considering model {args.which} using compressor {compressor_name}')
    logging.info(f'Model information: {config}')
    logging.info(f'Scale = {scale}, Offset = {offset}')

    data = get_data.fetch(stream_mode=stream_mode,
                          amt=int(args.amt),
                          context_window=cw,
                          filename=int(args.file),
                          scale=scale,
                          offset=offset,
                          map_fn=None,
                          rchunks=rchunks,
                          # If you want
                          random_order = False,
                          #random_order=True if not stream_mode else False,
                          noise = float(args.noise) if args.noise else None,
                          )

    mask_fn = None
    if info['vocab_size'] == 128:
        mask_fn = utils.right_shift_bytes_by_one

    rate, time, _ = compressor.evaluate_compressor(compress_fn_name=compressor_name,
                                                params=args.which,
                                                config=config,
                                                data=data,
                                                mask_fn=mask_fn,
                                                sigma = sigma,
                                                # mask_fn = None, #TEMP
                                                )

    logger.info(f'Compressor ran with {rate=} and {time=}')


def go():

    logger.info("Starting new run")

    # Fetch data
    n_chunks = 1
    cw = 512
    filename = 0
    compressor_name = 'btransformer'
    data = get_data.fetch(n_chunks, cw, filename)

    # Training
#    params, last_loss = btransformer.train.train_transformer_decoder(data = data,
    # tconfig = constants.TRANSFORMER_CONFIG,
    # training_steps = 20000,
    # log_every = 1000,
    # batch_size = 1,
    # learning_rate = 1e-5,
    # use_tqdm = True,
    # )

    # logger.info(f'{last_loss=}')
    # np.savez('btransformer/params.npz', **params)
    # logging.info('Parameters saved in file btransformer/params.npz')

    # Compress data
    test_data = []
    idx = 0
    for datum in data:
        test_data.append(datum)
        idx += 1
        if idx == 1:
            break

    plt.figure()
    plt.plot(np.frombuffer(test_data[0], dtype=np.uint8))
    plt.savefig('foobar.png')

    rate, time = compressor.evaluate_compressor(
        compressor_name, test_data, None, len(test_data), cw)
    logger.info(f'Compressor ran with {rate=} and {time=}')


if __name__ == '__main__':

    if args.action == 'train':
        train()
    elif args.action == 'compress':
        compress()
    elif args.action == 'decompress':
        decompress()
    elif args.action == 'eval':
        eval()
    else:
        logger.error('Not a valid action')

    logger.debug('Programme exiting')
