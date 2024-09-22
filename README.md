# Prompt Tuning on Graph-augmented Low-resource Text Classification
We provide the implementation of G2P2 and G2P2* model, which is the source code for the TKDE journal

"Prompt Tuning on Graph-augmented Low-resource Text Classification", and the link is https://ieeexplore.ieee.org/abstract/document/10633805. 

The repository is organised as follows:
- dataset/: the directory of data sets. Currently, it only has the dataset of Cora, if you want the three processed Amazon datasets, you can download and put them under this directory, the link is https://drive.google.com/drive/folders/1IzuYNIYDxr63GteBKeva-8KnAIhvqjMZ?usp=sharing.
- res/: the directory of saved models.
- meta_net/: the directoty of our G2P2* model
    - task_cora.py, task_amazon.py: data preprocessing for cora dataset and Amazon datasets
    - 	model_cocoop: model of G2P2*
    - 	main_cog2p2_cora.py, main_cog2p2_amazon.py: tuning and testing entrance for cora, tuning and testing entrance for Amazon datasets
- bpe_simple_vocab_16e6.txt.gz: vocabulary for simple tokenization.
- data.py, data_graph.py: for data loading utilization.
- main_test.py, main_test_amazon.py: testing entrance for cora, testing entrance for Amazon datasets.
- main_train.py, main_train_amazon.py: pre-training entrance for cora, pre-training entrance for Amazon datasets.
- model.py, model_g_coop.py: model for pre-training, model for prompt tuning.
- multitask.py, multitask_amazon.py: task generator for cora, task generator for Amazon datasets.
- requirements.txt: the required packages.
- simple_tokenizer: a simple tokenizer.


# For pre-train:
On Cora dataset,

    python main_train.py 

If on Amazon datasets, it should be:

    python main_train_amazon.py

# For testing:
(1) For G2P2,
On Cora dataset,

    python main_test.py 

If on Amazon datasets, it should be:

    python main_test_amazon.py
    
(2) For G2P2*,

    cd meta_net

On Cora dataset,

    python main_cog2p2_cora.py 

If on Amazon datasets, it should be:

    python main_cog2p2_amazon.py
    

