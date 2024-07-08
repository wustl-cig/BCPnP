import os
import pytorch_lightning as pl
from network.models.network_unet import UNetRes


def get_save_path_from_config(config):
    save_path = os.path.join(config['setting']['root_path'], config['setting']['save_path'])
    return save_path


def get_save_name_from_config(config):
    save_path = get_save_path_from_config(config)
    save_name = save_path.replace(config['setting']['root_path'], '').replace('ray_tune_run/ray_tune_', '').replace("/", "")

    return save_name


def get_trainer_from_config(config):

    save_path = get_save_path_from_config(config)

    callback = []

    if config['setting']['mode'] == 'tst' or config['setting']['mode'] == 'dug':

        trainer = pl.Trainer(
            accelerator='gpu',
            default_root_dir=save_path,
            log_every_n_steps=10,
            # strategy="ddp",
            callbacks=callback,
            inference_mode=False
        )

    else:

        raise ValueError()

    return trainer


def get_module_from_config(config, type_='x', use_sigma_map=False):
    assert type_ in ['x', 'theta']

    if type_ == 'x':
        DRUNET_nc = config['module']['gs_denoiser']['DRUNET_nc_x']
       
    else:
        DRUNET_nc = config['module']['gs_denoiser']['DRUNET_nc_cal']

    if 'pmri' in config['setting']['dataset']:
        nc = 2

        if config['module']['unetres']['pmri_num_coils'] > 0 and type_ != 'x':
            nc *= config['module']['unetres']['pmri_num_coils']

    else:

        if config['dataset']['natural']['subset'] in ['CBSD68'] and type_ == 'x': 
            nc = 3
        else:
            nc = 1

    module_dict = {

        'unetres': lambda: UNetRes(
            in_nc=nc + 1 if use_sigma_map else nc,
            out_nc=nc,
            nc=[DRUNET_nc, DRUNET_nc * 2, DRUNET_nc * 4, DRUNET_nc * 8],
            nb=4,
            act_mode='R',
            downsample_mode='strideconv',
            upsample_mode='convtranspose'),

    }

    return module_dict[config['method']['bcpnp'][type_ + '_module']]
