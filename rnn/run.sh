python main.py --cuda_visible_devices '1' --config_path './config.yml' --load_wv_path '../word2vec/skip_gram.wv' --batch_size 128 --train_steps 20000 --log_file 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_attention_size_128_steps_20000.log' --checkpoint_dir 'embed_128_hidden_256_layer_3_pretrain_wv_dropout_0.5_attention_size_128_steps_20000/'


python main.py --cuda_visible_devices '0' --config_path './config.yml' --load_wv_path '../word2vec/skip_gram.wv' --batch_size 128 --train_steps 20000 --is_demo 1