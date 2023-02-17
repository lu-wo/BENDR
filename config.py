

params = {}

params['epochs'] = 10
params['in_features'] = 128
params['window_len'] = 1000
#Dataset
params['root_dir'] = '/itet-stor/ljie/deepeye_itetnas04'
params['batch_size'] = 32
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
params['encoder_h'] =256
params['enc_width'] = (3, 2, 2, 2, 2, 2)
params['enc_dropout'] = 0.0
params['projection_head'] = False
params['enc_downsample'] = (3, 2, 2, 2, 2, 2)

#BENDR
params['mask_rate'] = 0.1
params['mask_span'] = 6
params['learning_rate'] = 0.01 
params['temp'] = 0.5
params['permuted_encodings'] = False
params['permuted_contexts'] = False
params['enc_feat_l2'] = 0.001
params['multi_gpu'] = False
params['l2_weight_decay'] = 1e-4
params['unmasked_negative_frac'] = 0.25
params['encoder_grad_frac'] = 1.0
params['num_negatives'] = 100