python main.py --cuda_visible_devices '0' --optimizer 'SGD' --lr 0.1 --train_steps 30000 --log_file 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_30000_sgd.log' --checkpoint_dir 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_30000_sgd/'


python main.py --cuda_visible_devices '1' --train_steps 1000 --optimizer 'SGD' --lr 1 --momentum 0.9
python main.py --cuda_visible_devices '2' --train_steps 1000 --optimizer 'SGD' --lr 0.1 --momentum 0.9

python main.py --cuda_visible_devices '1' --optimizer 'SGD' --lr 0.1 --momentum 0.9 --train_steps 30000 --log_file 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_30000_sgd_momentum.log' --checkpoint_dir 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_30000_sgd_momentum/'


python main.py --cuda_visible_devices '0' --optimizer 'SGD' --lr 0.1 --momentum 0.9 --lr_scheduler 'step' --train_steps 20000 --log_file 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_20000_sgd_momentum_lr_step.log' --checkpoint_dir 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_20000_sgd_momentum_lr_step/'
python main.py --cuda_visible_devices '1' --optimizer 'SGD' --lr 0.1 --momentum 0.9 --lr_scheduler 'metric' --train_steps 20000 --log_file 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_20000_sgd_momentum_lr_metric.log' --checkpoint_dir 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_20000_sgd_momentum_lr_metric/'
python main.py --cuda_visible_devices '2' --optimizer 'SGD' --lr 0.1 --lr_scheduler 'step' --lr_step_gamma 0.75 --train_steps 20000 --log_file 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_20000_sgd_lr_step.log' --checkpoint_dir 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_maxpooling_steps_20000_sgd_lr_step/'


python main.py --cuda_visible_devices '0' --optimizer 'SGD' --lr 0.1 --momentum 0.9 --lr_scheduler 'step' --lr_step_size 3 --train_steps 1000 --is_demo 1
python main.py --cuda_visible_devices '1' --optimizer 'SGD' --lr 0.1 --momentum 0.9 --lr_scheduler 'metric' --lr_patience 2 --train_steps 1000 --is_demo 1