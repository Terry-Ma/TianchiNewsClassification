CUDA_VISIBLE_DEVICES='0' python ../main.py --experiment_name 'gru_w2v_dropout_maxpooling_baseline'

CUDA_VISIBLE_DEVICES='0' python ../main.py --experiment_name 'gru_baseline_add_embed_spatial_dropout_v2' --load_checkpoint_path '/home/luxiaoling/mayunquan/TianchiNewsClassification/rnn/checkpoint/gru_baseline_add_embed_spatial_dropout/best_model.cpt' --train_steps 10000 --use_embed_dropout 1 --embed_dropout_type 'spatial'

CUDA_VISIBLE_DEVICES='1' python ../main.py --experiment_name 'gru_baseline_add_embed_dropout_embed' --use_embed_dropout 1 --embed_dropout_type ''

CUDA_VISIBLE_DEVICES='0' python ../main.py --experiment_name 'gru_baseline_add_embed_spatial_dropout_v2' --use_embed_dropout 1