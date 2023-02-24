
params = {}

# BENDR pretrained
params['bendr_dir'] = '/itet-stor/wolflu/net_scratch/projects/BENDR/reports/logs/20230224-133833'

# Finetuning 
params['data_dir'] = '/itet-stor/wolflu/deepeye_itetnas04/semester-project-djl/datasets/LR_task_with_antisaccade_synchronised_min.npz'
params['epochs'] = 10
params['batch_size'] = 32
params['learning_rate'] = 1e-5
params['loss'] = 'cross_entropy' # 'cross_entropy' or 'mse'

# Finetuning model 
params['hidden_layers'] = 2
params['output_logits'] = 1 
params['hidden_size'] = 32
params['dropout'] = 0.1

