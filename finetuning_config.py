
params = {}

# BENDR pretrained
params['bendr_dir'] = "/itet-stor/klebed/deepeye_itetnas04/semester-project-djl/pretrained_bendr/20230224-151256"
params['task'] = 'position' # 'position' or 'LR

# Finetuning 
params['data_dir'] = '/itet-stor/klebed/deepeye_itetnas04/semester-project-djl/datasets/Position_task_with_dots_synchronised_min.npz' 
params['epochs'] = 20
params['batch_size'] = 64
params['learning_rate'] = 1e-4 
params['loss'] = 'mse' # 'cross_entropy' or 'mse'

# Finetuning model 
params['hidden_layers'] = 2
params['output_logits'] = 2 
params['hidden_size'] = 32
params['dropout'] = 0.1

