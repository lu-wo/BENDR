
params = {}

# Training 
params['epochs'] = 200 
params['multi_gpu'] = False
# params['gpus'] = 2
params['learning_rate'] = 1e-5

#Dataset and loader
params['dataset'] = 'all' # 'all' or 'dots' or 'movie' or 'zuco' or 'processing_speed' or 'debug'

if params['dataset'] == 'debug':
    params['root_dir'] = '/itet-stor/wolflu/net_scratch/projects/EEGEyeNet_experimental/data/stream_debug' 
elif params['dataset'] == 'all':
    params['root_dir'] = '/itet-stor/wolflu/deepeye_itetnas04/data/single_stream'
elif params['dataset'] == 'movie':
    params['root_dir'] = '/itet-stor/wolflu/deepeye_itetnas04/data/single_stream/movie'
elif params['dataset'] == 'zuco':
    params['root_dir'] = '/itet-stor/wolflu/deepeye_itetnas04/data/single_stream/zuco'
elif params['dataset'] == 'processing_speed':
    params['root_dir'] = '/itet-stor/wolflu/deepeye_itetnas04/data/single_stream/processing_speed'
elif params['dataset'] == 'dots':
    params['root_dir'] =  '/itet-stor/wolflu/deepeye_itetnas04/data/single_stream/dots' 

params['batch_size'] = 16
params['window_len'] = 4000

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
params['encoder_h'] = 128
params['enc_width'] = (3, 2, 2, 2) # 2000: 3x2, 4000: 4x2, 8000: 5x2
params['enc_downsample'] = (3, 2, 2, 2)
params['enc_dropout'] = 0.0
params['projection_head'] = False

#BENDR
params['mask_rate'] = 0.1
params['mask_span'] = 5
params['temp'] = 1
params['permuted_encodings'] = False
params['permuted_contexts'] = False
params['enc_feat_l2'] = 0.001
params['l2_weight_decay'] = 1e-4
params['unmasked_negative_frac'] = 0.25
params['encoder_grad_frac'] = 1.0
params['num_negatives'] = 20
