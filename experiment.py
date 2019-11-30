from sacred import Experiment
from sacred.observers import MongoObserver
from datetime import datetime
import os

experiment_name = 'validate_adversarial_approach'
experiment_name += '_'+datetime.now().strftime('%y%m%d_%H%M%S')

exp = Experiment(name=experiment_name)
exp.observers.append(MongoObserver.create())

config_model_folder = experiment_name
base_model_folder = '/path/to/model_folder'

exp.add_config(
{
    'n_layers_ar': 7,
    'n_layers_fft': 8,
    'time_distributed': False,
    'adversarial_strength': 1.25,
    'mode': 'conditional',
    'delay_adversary_epochs': 50,
    'earlystop': 0,
    'nr_epochs': 1000,
    'batch_size': 16,
    'model_dir': os.path.join(base_model_folder, config_model_folder)
})


@exp.automain
def run(_run, n_layers_ar, n_layers_fft, time_distributed, adversarial_strength,
        mode, delay_adversary_epochs, earlystop, nr_epochs, batch_size, model_dir, ppg):
    import numpy as np

    architecture = Architecture(n_layers_ar=n_layers_ar, n_layers_fft=n_layers_fft,
                                time_distributed=time_distributed, adversarial_strength=adversarial_strength)


    model = AdversarialModel(input_shapes=x_train.shape, output_shape=y_train.shape,
                             domain_shape=d_train.shape, architecture=architecture, model_dir=model_dir,
                             mode=mode, delay_adversary_epochs=delay_adversary_epochs, sacred_object=_run)

    _run.info['classifier_params'] = model.classifier_model.count_params()

    try:
        model.train_adversarial(x_train=x_train, y_train=y_train, d_train=d_train, w_train=sample_weights,
                                             mask_train=train_mask, x_test=x_test, , y_test=y_test, mask_test=test_mask,
                                             nr_epochs=nr_epochs, earlystop=earlystop, batch_size=batch_size)

        stats = calc_stats_auc(model.classifier_model, x_test, y_test, test_mask, thr=0.5, color='b')
    except Exception as e:
        _run.info['Error'] = e
        return -1
    for key in stats.keys():
        _run.info[key] = stats[key]

    return stats['acc']


