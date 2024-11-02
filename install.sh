conda create -n llmcomp python=3.12
conda activate llmcomp

# pip packages go here
pip install torch
pip install transformers
pip install SentencePiece
pip install accelerate
pip install numpy
pip install pandas
pip install audioop
pip install absl
pip install jax[cuda12]
pip install tqdm
pip install matplotlib
pip install dm-haiku
pip install dm-tree
pip install tensorflow

conda deactivate