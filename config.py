
params = {}

# Training 
params['epochs'] = 100
# params['multi_gpu'] = True
params['gpus'] = 4
params['learning_rate'] = 1e-5

#Dataset and loader
# params['root_dir'] = '/itet-stor/wolflu/net_scratch/projects/EEGEyeNet_experimental/data/stream_debug' # '/itet-stor/wolflu/net_scratch/projects/EEGEyeNet_experimental/data/stream_debug' # '/itet-stor/wolflu/deepeye_itetnas04/data/single_stream' # change to your path
params['root_dir'] =  '/itet-stor/wolflu/deepeye_itetnas04/data/single_stream' 
params['batch_size'] = 64
params['window_len'] = 2000

#Contextualizer
params['hidden_feedforward'] = 3076
params['heads'] = 8
params['layers'] = 8
params['dropout'] = 0.15
params['activation'] = 'gelu'
params['position_encoder'] = 25
params['layer_drop'] = 0.0
params['mask_p_t'] = 0.1
params['mask_p_c'] = 0.004
params['mask_t_span'] = 6
params['mask_c_span'] = 64
params['start_token'] = -5
params['finetuning'] = False

#Encoder
params['in_features'] = 129
params['encoder_h'] = 256
params['enc_width'] = (3, 2, 2, 2, 2)
params['enc_downsample'] = (3, 2, 2, 2, 2)
params['enc_dropout'] = 0.0
params['projection_head'] = False

#BENDR
params['mask_rate'] = 0.1 
params['mask_span'] = 5
params['temp'] = 0.5
params['permuted_encodings'] = False
params['permuted_contexts'] = False
params['enc_feat_l2'] = 0.001
params['l2_weight_decay'] = 1e-4
params['unmasked_negative_frac'] = 0.25
params['encoder_grad_frac'] = 1.0
params['num_negatives'] = 100
