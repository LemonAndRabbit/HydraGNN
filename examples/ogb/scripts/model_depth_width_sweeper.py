import json
import os
import copy

exp_name = 'exp_ogb_gap_model_depth_width_sweep'
cur_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
base_config = os.path.join(cur_path, 'ogb_gap.json')
python_script = os.path.join(cur_path, 'train_gap.py')
dataset_folder = os.path.join(cur_path, 'dataset')
with open(base_config, 'r') as f:
    base_config = json.load(f)

for num_conv_layers in [6, 10, 12]:
    for hidden_dim in [64, 128, 256, 512]:
        
        # create temporary running folder
        exp_folder = f'{exp_name}/l_{num_conv_layers}_h{hidden_dim}'
        exp_folder = os.path.join(cur_path, exp_folder)
        os.makedirs(exp_folder, exist_ok=True)
        
        # save updated config
        config = copy.deepcopy(base_config)
        config['NeuralNetwork']['Architecture']['hidden_dim'] = hidden_dim
        config['NeuralNetwork']['Architecture']['num_conv_layers'] = num_conv_layers
        with open(f'{exp_folder}/ogb_gap.json', 'w') as f:
            json.dump(config, f)

        # soft link the python script and dataset folder
        os.system(f'ln -s {python_script} {exp_folder}/train_gap.py')
        os.system(f'ln -s {dataset_folder} {exp_folder}/dataset')

        # enter the folder and run experiment
        os.chdir(exp_folder)
        os.system(f'mpirun -n 8 python train_gap.py --pickle gap --mae')

        # remove the soft links
        os.system(f'rm train_gap.py')
        os.system(f'rm dataset')