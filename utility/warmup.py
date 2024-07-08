import torch
import os


warmup_list = {

    'pmri_fastmri': {
        'x': {
                'unetres': {
                    'g_denoise': 'image_denoiser.ckpt',
                }
        },

        'theta':
            {
                'unetres': {
                    'g_denoise': 'theta_denoiser.ckpt',
                }
            }
    },

}


def load_warmup(
        target_module,
        dataset,
        gt_type,
        pattern,
        sigma,
        prefix,
        is_print=True,
        network='unet',
        is_load_state_dict=True
):

    if pattern == 'denoise':
        if isinstance(sigma, int):
            x_ckpt = warmup_list[dataset][gt_type][network]["%s_%d" % (pattern, sigma)]
        else:
            x_ckpt = warmup_list[dataset][gt_type][network]["%s_%.2f" % (pattern, sigma)]
    else:
        x_ckpt = warmup_list[dataset][gt_type][network][pattern]

    if is_print:
        print("Loading ckpt from", x_ckpt)

    if is_load_state_dict:
        x_ckpt = torch.load(os.path.join('network', x_ckpt))['state_dict']
    else:
        x_ckpt = torch.load(os.path.join('network', x_ckpt))

    x_self = target_module.state_dict()

    for name, param in x_self.items():

        name_ckpt = prefix + name

        if name_ckpt not in x_ckpt:
            raise ValueError('cannot find %s in the checkpoint' % name_ckpt)

        param_ckpt = x_ckpt[name_ckpt]
        if isinstance(param_ckpt, torch.nn.parameter.Parameter):
           
            param_ckpt = param_ckpt.data

        x_self[name].copy_(param_ckpt)
