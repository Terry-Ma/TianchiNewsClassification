CUDA_VISIBLE_DEVICES='0' python ../main.py --num_hidden_layers 4 --hidden_size 256 --num_attention_heads 4 --intermediate_size 1024 --batch_size 32 --lr 0.00001 --warmup_steps 10000 --train_steps 50000 --experiment_name 'bert_mini_lr_1e-5_warmup_steps_10000_train_steps_50000'

CUDA_VISIBLE_DEVICES='1' python ../main.py --num_hidden_layers 4 --hidden_size 256 --num_attention_heads 4 --intermediate_size 1024 --batch_size 32 --lr 0.0001 --warmup_steps 10000 --train_steps 50000 --experiment_name 'bert_mini_lr_1e-4_warmup_steps_10000_train_steps_50000'

CUDA_VISIBLE_DEVICES='0' python ../main.py --batch_size 32 --experiment_name 'bert_mini_gru_lr_1e-5_batch_size_32_warmup_steps_10000_train_steps_80000'

CUDA_VISIBLE_DEVICES='1' python ../main.py --batch_size 32 --agg_function 'Attention' --experiment_name 'bert_mini_attention_lr_1e-5_batch_size_32_warmup_steps_10000_train_steps_80000'