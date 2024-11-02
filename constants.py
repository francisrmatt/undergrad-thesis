"""Project wide default values"""
from btransformer.transformer import TransformerConfig
#from llama.llama import LlamaConfig
# Do logging levels
import os
LOG_LEVEL = os.environ.get('LLMCOMP_LOG_LEVEL', 'INFO')

LOGGING_CONFIG = {
    'version' : 1,
    'formatters' : 
        {'standard' : {
            'format' : '%(asctime)s %(name)s [%(levelname)s] %(message)s'},
        },
    'handlers' : 
        {'console' : {
            'class' : 'logging.StreamHandler',
            'formatter' : 'standard',
            'level' : LOG_LEVEL,
        },
         'file' : {
             'class' : 'logging.FileHandler',
             'formatter' : 'standard',
             'filename' : 'debug.log',
             'level' : LOG_LEVEL,
         }},
    'loggers' : {
        '' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'compressor' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'get_data' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'language_model' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        # Add as needed
        'btransformer.train' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
        'llama.compress' : {
            'handlers' : ['console', 'file'],
            'level' : LOG_LEVEL,
            'propagate' : False,
        },
    }
}
TRANSFORMER_CONFIG = TransformerConfig(
        vocab_size = 128,
        embedding_dim = 128,
        num_heads = 8,
        num_layers = 16,
        emb_init_scale = 0.02,
        widening_factor = 4,
)

#LLAMA_CONFIG = LlamaConfig()

CODER_PRECISION = 32

SNR_SD = [#8879.514423764482,
 #4993.317908106538,
 2807.949009541332,
 887.9514423764482,
 280.79490095413325,
 88.79514423764483,
 None,
 #28.079490095413323
 ]

SCALES = [
    1.1,
    1,
    0.9, 
    0.8, 
    0.7
]

SNR_SET = [
    #0,
    #5,
    10,
    20,
    30,
    40,
    999,
    #50,
]

NUM_TEST_FILES = 10

GZIP_RESULTS = [0.87295703125, 0.8592929687499999, 0.85558984375, 0.8105, 0.64748828125]
AC_RESULTS = [0.870420979299363, 0.8584569068471337, 0.8572835390127389, 0.8571501791401274, 0.8571342555732485]


import pandas as pd
data = {
    1.1: [0.889465, 0.876227, 0.872854, 0.829555, 0.653707],
    1.0: [0.871733, 0.858504, 0.854635, 0.806607, 0.640636],
    0.9: [0.852040, 0.838982, 0.834540, 0.781086, 0.626396],
    0.8: [0.830539, 0.818205, 0.812930, 0.753218, 0.611267],
    0.7: [0.807696, 0.796103, 0.789399, 0.724633, 0.595175]
}

row_names = [10, 20, 30, 40, 999]
GZIP_RESULTS= pd.DataFrame(data, index=row_names)

data = {
    1.1: [2.548167e+03, 8.058013e+02, 2.548167e+02, 8.058013e+01, 9.041239e-47],
    1.0: [2.316619e+03, 7.325792e+02, 2.316619e+02, 7.325792e+01, 8.219674e-47],
    0.9: [2.084843e+03, 6.592854e+02, 2.084843e+02, 6.592854e+01, 7.397304e-47],
    0.8: [1.853194e+03, 5.860315e+02, 1.853194e+02, 5.860315e+01, 6.575381e-47],
    0.7: [1.621515e+03, 5.127680e+02, 1.621515e+02, 5.127680e+01, 5.753352e-47]
}
row_names_new = [10, 20, 30, 40, 999]
SNR_SD_TABLE = pd.DataFrame(data, index=row_names_new)

data = {
    1.1: [0.889262, 0.880220, 0.879309, 0.879265, 0.879274],
    1.0: [0.870696, 0.862948, 0.862162, 0.862122, 0.862097],
    0.9: [0.851791, 0.845606, 0.845004, 0.844914, 0.844909],
    0.8: [0.835325, 0.830399, 0.829882, 0.829852, 0.829857],
    0.7: [0.820785, 0.816974, 0.816595, 0.816561, 0.816531],
}
ac_row_names =  [10, 20, 30, 40, 999]
AC_RESULTS = pd.DataFrame(data, index = ac_row_names)
