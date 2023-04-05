
params = {}

# BENDR pretrained
params['bendr_dir'] = '/itet-stor/wolflu/net_scratch/projects/BENDR/reports/logs/20230327-185858'
params['task'] = 'amplitude' # 'position' or 'LR angle' or 'angle' or 'ampltiude'

# Finetuning 
params['data_dir'] = '/itet-stor/wolflu/deepeye_itetnas04/semester-project-djl/datasets/Direction_task_with_dots_minprep_synch.npz' 
params['epochs'] = 10
params['batch_size'] = 32
params['learning_rate'] = 1e-3
params['loss'] = 'mse' # 'cross_entropy' or 'mse' or 'angle_loss'

# Finetuning model 
params['hidden_layers'] = 4
params['output_logits'] = 1 
params['hidden_size'] = 64
params['dropout'] = 0.1
