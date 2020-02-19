#!/bin/bash

set -x                                           

for i; do
    echo $i
done

nvidia-smi

sleep 5

export CUDA_VISIBLE_DEVICES=0                                        #export gpu till here

DATAHOME=${@:(-2):1}                                                 #save adress of workspace>neusum>data
EXEHOME=${@:(-1):1}                                                  #save adress of workspace>neusum>code

ls -l ${DATAHOME}

ls -l ${EXEHOME}

SAVEPATH=${DATAHOME}/models/neusum                                   #reach to the directory workspace>neusum>data>models>neusum

mkdir -p ${SAVEPATH}                                                 #and there make a new folder

cd ${EXEHOME}

# python train.py -save_path ${SAVEPATH} \
#                 -online_process_data \
#                 -max_doc_len 80 \
#                 -train_oracle ${DATAHOME}/train/train.rouge_bigram_F1.oracle.F1mmrTrue.regGain \
#                 -train_src ${DATAHOME}/train/train.txt.src \
#                 -train_src_rouge ${DATAHOME}/train/train.rouge_bigram_F1.oracle.F1mmrTrue.regGain \
#                 -src_vocab ${DATAHOME}/train/vocab.txt.100k \
#                 -train_tgt ${DATAHOME}/train/train.txt.tgt \
#                 -tgt_vocab ${DATAHOME}/train/vocab.txt.100k \
#                 -layers 1 -word_vec_size 50 -sent_enc_size 256 -doc_enc_size 256 -dec_rnn_size 256 \
#                 -sent_brnn -doc_brnn \
#                 -dec_init simple \
#                 -att_vec_size 256 \
#                 -norm_lambda 20 \
#                 -sent_dropout 0.3 -doc_dropout 0.2 -dec_dropout 0\
#                 -batch_size 64 -beam_size 1 \
#                 -epochs 100 \
#                 -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
#                 -gpus 0 \
#                 -curriculum 0 -extra_shuffle \
#                 -start_eval_batch 1000 -eval_per_batch 1000 \
#                 -log_interval 100 -log_home ${SAVEPATH} \
#                 -seed 12345 -cuda_seed 12345 \
#                 -pre_word_vecs_enc ${DATAHOME}/glove/glove.6B.50d.txt \
#                 -freeze_word_vecs_enc \
#                 -dev_input_src ${DATAHOME}/dev/val.txt.src.shuffle.4k \
#                 -dev_ref ${DATAHOME}/dev/val.txt.tgt.shuffle.4k \
#                 -max_decode_step 3 -force_max_len





#in the below line train.py python script is being callled with all the required variables in it where we set document length is equal to 80 word vector size is 50 
#encoder size is 256 same as decoder size also attention vector msize is 256 dropout for sentence level encoding is 0.3 while document level encoding is 0.2
#maximum epoch ase set to 100 and in each epoch there are 1000 batches and batch size is 64  adam is used as optimizer with learning rate 0.001
#a log file having information about process is saved after 100 batches at last maximum decoding limit is set to be 3 sent and train and validation's source ans target file are retrived.

python train.py -save_path ${SAVEPATH} \
                -online_process_data \
                -max_doc_len 80 \
                -train_oracle ${DATAHOME}/train/oracle_regain_output.txt \
                -train_src ${DATAHOME}/train/train_src.txt \
                -train_src_rouge ${DATAHOME}/train/oracle_regain_output.txt \
                -src_vocab ${DATAHOME}/train/vocab_tgt.txt \
                -train_tgt ${DATAHOME}/train/train_tgt.txt \
                -tgt_vocab ${DATAHOME}/train/vocab_tgt.txt \
                -layers 1 -word_vec_size 50 -sent_enc_size 256 -doc_enc_size 256 -dec_rnn_size 256 \
                -sent_brnn -doc_brnn \
                -dec_init simple \
                -att_vec_size 256 \
                -norm_lambda 20 \
                -sent_dropout 0.3 -doc_dropout 0.2 -dec_dropout 0\
                -batch_size 64 -beam_size 1 \
                -epochs 100 \
                -optim adam -learning_rate 0.001 -halve_lr_bad_count 100000 \
                -gpus 0 \
                -curriculum 0 -extra_shuffle \
                -start_eval_batch 1000 -eval_per_batch 1000 \
                -log_interval 100 -log_home ${SAVEPATH} \
                -seed 12345 -cuda_seed 12345 \
                -pre_word_vecs_enc ${DATAHOME}/glove/glove.6B.50d.txt \
                -freeze_word_vecs_enc \
                -dev_input_src ${DATAHOME}/dev/val_src_4k.txt \
                -dev_ref ${DATAHOME}/dev/val_tgt_4k.txt \
                -max_decode_step 3 -force_max_len