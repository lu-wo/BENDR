
params = {}

# BENDR pretrained
params['bendr_dir'] = "/itet-stor/klebed/deepeye_itetnas04/semester-project-djl/pretrained_bendr/20230224-151256"

# Finetuning 
params['data_dir'] = '/itet-stor/klebed/deepeye_itetnas04/semester-project-djl/datasets/LR_task_debug.npz' 
params['epochs'] = 10
params['batch_size'] = 32
params['learning_rate'] = 2e-4 
params['loss'] = 'cross_entropy' # 'cross_entropy' or 'mse'

# Finetuning model 
params['hidden_layers'] = 2
params['output_logits'] = 1 
params['hidden_size'] = 32
params['dropout'] = 0.1

