params = {}

# Training
params["epochs"] = 250
params["multi_gpu"] = False
# params['gpus'] = 2
params["learning_rate"] = 1e-5

# Dataset and loader
params[
    "dataset"
] = "all"  # 'all' or 'dots' or 'movie' or 'zuco' or 'processing_speed' or 'debug'

params[
    "data_path"
] = "/Users/lukas/Desktop/projects/deepeye_all/BENDR_emg/data/01_raw/mat"  # "/nese/mit/group/evlab/u/luwo/projects/projects/semidetr/SemiDETRtime/data/01_raw"

params["batch_size"] = 32
# params["window_len"] = 2000

# Contextualizer
params["hidden_feedforward"] = 2048
params["heads"] = 8
params["layers"] = 8
params["dropout"] = 0.15
params["activation"] = "gelu"
params["position_encoder"] = 25
params["layer_drop"] = 0.0
params["mask_p_t"] = 0.1
params["mask_p_c"] = 0.004
params["mask_t_span"] = 6
params["mask_c_span"] = 64
params["start_token"] = -5
params["finetuning"] = False

# Encoder
params["in_features"] = 16
params["encoder_h"] = 512
params["enc_width"] = (16, 8, 8, 3, 3, 3)  # 2000: 3x2, 4000: 4x2, 8000: 5x2
params["enc_downsample"] = (5, 4, 2, 2, 1, 1)
params["enc_dropout"] = 0.1
params["projection_head"] = False

# BENDR
params["mask_rate"] = 0.1
params["mask_span"] = 5
params["temp"] = 0.5
params["permuted_encodings"] = False
params["permuted_contexts"] = False
params["enc_feat_l2"] = 0.001
params["l2_weight_decay"] = 1e-4
params["unmasked_negative_frac"] = 0.25
params["encoder_grad_frac"] = 1.0
params["num_negatives"] = 20
