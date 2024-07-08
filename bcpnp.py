import glob
from torch import nn
import pytorch_lightning as pl
from torchmetrics.functional.image import structural_similarity_index_measure, peak_signal_noise_ratio
from utility.warmup import load_warmup
import torch
from utility.get_from_config import get_trainer_from_config, get_save_path_from_config,  get_module_from_config
from torch.utils.data import DataLoader
from einops import rearrange
from utility.helper import convert_pl_outputs, check_and_mkdir, write_test
import pickle
import matplotlib.pyplot as plt
from skimage.io import imsave
import os
import numpy as np
from skimage.metrics import structural_similarity


def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """

    y = y * mask.unsqueeze(1)

    y = torch.fft.ifftshift(y, [-2, -1])
    x = torch.fft.ifft2(y, norm='ortho')
    x = torch.fft.fftshift(x, [-2, -1])

    x = x * torch.conj(smps)
    x = x.sum(1)

    return x


def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """
   
    x = x.unsqueeze(1)
    y = x * smps

    y = torch.fft.ifftshift(y, [-2, -1])
    y = torch.fft.fft2(y, norm='ortho')
    y = torch.fft.fftshift(y, [-2, -1])

    mask = mask.unsqueeze(1)
    y = y * mask

    return y


def divided_by_rss(smps):
    return smps / (torch.sum(torch.abs(smps) ** 2, 1, keepdim=True).sqrt() + 1e-10)


def gradient_smps(smps, x, y, mask):
    x_adjoint = torch.conj(x)

    ret = fmult(x, smps, mask) - y
    ret = ret * mask.unsqueeze(1)

    ret = torch.fft.ifftshift(ret, [-2, -1])
    ret = torch.fft.ifft2(ret, norm='ortho')
    ret = torch.fft.fftshift(ret, [-2, -1])

    ret = ret * x_adjoint.unsqueeze(1)

    return ret


class ImageUpdate(nn.Module):
    def __init__(self, net_x, config):
        super().__init__()

        self.config = config
        self.cnn = net_x()

        self.gamma = 1.5
        self.alpha = 0.05

        self.sigma = config['method']['bcpnp']['warmup']['x_sigma'] / 255

    def denoise_complex(self, x, sigma=None):

        x_hat = torch.view_as_real(x)
        x_hat = rearrange(x_hat, 'b w h c -> b c w h')

        pad = 0
        if x_hat.shape[-1] == 396:
            pad = 4

        if pad > 0:
            x_hat = torch.nn.functional.pad(x_hat, [pad, 0])

        if sigma is not None:
            noise_level_map = torch.FloatTensor(x_hat.size(0), 1, x_hat.size(2), x_hat.size(3)).fill_(sigma).to(x_hat.device)
            x_hat = torch.cat((x_hat, noise_level_map), 1)
            x_hat = self.cnn(x_hat)

        else:
            x_hat = self.cnn(x_hat)

        if pad > 0:
            x_hat = x_hat[..., pad:]

        x_hat = rearrange(x_hat, 'b c w h -> b w h c')
        x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        return x_hat

    def forward(self, x, theta, mask, y):
        dc = ftran(fmult(x, theta, mask) - y, theta, mask)

        x = x - self.gamma * dc 

        if self.config['method']['bcpnp']['warmup']['x_ckpt'] == 'g_denoise':
            prior = self.denoise_complex(x, self.sigma)
        else:
            prior = self.denoise_complex(x)

        x_hat = self.alpha * prior + (1 - self.alpha) * x

        return x_hat


class ParameterUpdate(nn.Module):
    def __init__(self, net_theta, config):
        super().__init__()

        self.config = config
        self.cnn = net_theta()

        self.gamma = 1.5
        self.alpha = 0.05

        self.is_update_theta_iteratively = True
        self.is_update_theta_iteratively_bc = True

        self.sigma = config['method']['bcpnp']['warmup']['theta_sigma'] / 255

    def calibrate_complex(self, x, sigma=None):
        batch_size = x.shape[0]

        x = torch.view_as_real(x)

        if self.config['module']['unetres']['pmri_num_coils'] > 0:
            l = x.shape[1]
            x = rearrange(x, 'b l w h c -> b (l c) w h')

        else:
            x = rearrange(x, 'b l h w c -> (b l) c h w')

        pad = 0
        if x.shape[-1] == 396:
            pad = 4

        if pad > 0:
            x = torch.nn.functional.pad(x, [pad, 0])

        if sigma is not None:
            noise_level_map = torch.FloatTensor(x.size(0), 1, x.size(2), x.size(3)).fill_(sigma).to(
                x.device)
            x_hat = torch.cat((x, noise_level_map), 1)
            x_hat = self.cnn(x_hat)

        else:
            x_hat = self.cnn(x)

        if pad > 0:
            x_hat = x_hat[..., pad:]

        if self.config['module']['unetres']['pmri_num_coils'] > 0:
            x_hat = rearrange(x_hat, 'b (l c) w h -> b l w h c', l=l)
        else:
            x_hat = rearrange(x_hat, '(b l) c h w -> b l h w c', b=batch_size)

        x_hat = x_hat[..., 0] + x_hat[..., 1] * 1j

        return x_hat

    def forward(self, theta, x, mask, y, theta_label):
        if self.is_update_theta_iteratively:
            dc = gradient_smps(theta, x, y, mask)

            if self.is_update_theta_iteratively_bc:
                theta = theta - self.gamma * dc

                if self.config['method']['bcpnp']['warmup']['theta_ckpt'] == 'g_denoise':
                    prior = self.calibrate_complex(theta, self.sigma)
                else:
                    prior = self.calibrate_complex(theta)

                theta = self.alpha * prior + (1 - self.alpha) * theta

            else:
                prior = theta - theta_label
                theta = theta - self.gamma * (dc + self.alpha * prior)

            num_coil = theta.shape[1]

            rss = torch.sum(torch.abs(theta) ** 2, 1, keepdim=True).sqrt().repeat([1, num_coil, 1, 1])
            theta[rss > 1] /= rss[rss > 1]

        return theta


class GenericAccelerator:
    def __init__(self, x_init):
        self.t = 1.0
        self.x_prev = x_init

    def __call__(self, f, s, **kwargs):
        xnext = f(s, **kwargs)

        res = (xnext - self.x_prev).norm().item() / xnext.norm().item()

        self.x_prev = xnext

        return xnext, res


class BCPnP(pl.LightningModule):
    def __init__(self, net_x, net_theta, config):
        super().__init__()

        self.config = config
        self.iterations = -1

        self.x_operator = ImageUpdate(net_x, self.config)
        self.theta_operator = ParameterUpdate(net_theta, self.config)

        x_pattern = self.config['method']['bcpnp']['warmup']['x_ckpt']
        if x_pattern is not None:
            load_warmup(
                target_module=self.x_operator.cnn,
                dataset=self.config['setting']['dataset'],
                gt_type='x',
                pattern=x_pattern,
                sigma=self.config['method']['bcpnp']['warmup']['x_sigma'],
                prefix='net.',
                network=self.config['method']['bcpnp']['x_module']
            )

        theta_pattern = self.config['method']['bcpnp']['warmup']['theta_ckpt']
        if theta_pattern is not None:
            load_warmup(
                target_module=self.theta_operator.cnn,
                dataset=self.config['setting']['dataset'],
                gt_type='theta',
                pattern=theta_pattern,
                sigma=self.config['method']['bcpnp']['warmup']['theta_sigma'],
                prefix='net.',
                network=self.config['method']['bcpnp']['theta_module']
            )

        self.is_joint_cal = True

        self.accelerator_dict = {
            'generic': lambda x_init: GenericAccelerator(x_init),
        }
        self.accelerator = 'generic'

    def forward(self, x_init, theta_init, mask, y, x_gt, theta_gt):

        x_hat, theta_hat = x_init, theta_init
        theta_hat = divided_by_rss(theta_hat)

        theta_label = None

        print("Start iterations ...")
        if self.iterations == -1:

            max_iter = 500
            tol = 0.00001

            with torch.no_grad():

                x_pre, theta_pre = x_hat, theta_hat

                x_accelerator = self.accelerator_dict[self.accelerator](x_pre)
                theta_accelerator = self.accelerator_dict[self.accelerator](theta_pre)

                for forward_iter in range(max_iter):

                    update_idx = [0, 1]

                    for idx in update_idx:

                        if idx == 0:
                            x_hat, _ = x_accelerator(
                                self.x_operator.forward, x_pre,
                                theta=theta_hat, mask=mask, y=y
                            )

                        elif idx == 1:

                            if self.is_joint_cal:
                                theta_hat, _ = theta_accelerator(
                                    self.theta_operator.forward, theta_pre,
                                    x=x_hat, mask=mask, y=y, theta_label=theta_label
                                )
                            else:
                                theta_hat = theta_pre

                    forward_res_theta = torch.norm(theta_pre - theta_hat) ** 2
                    forward_res_x = torch.norm(x_pre - x_hat) ** 2

                    forward_res = forward_res_x + forward_res_theta

                    if torch.isnan(forward_res):
                        print("meet nan in the iteration")
                        exit(0)

                    if forward_res < tol:
                        break

                    x_pre = x_hat
                    theta_pre = theta_hat

                    if (forward_iter + 1) % 50 == 0:
                        print("Iteration [%d] Image Residual [%.5f] Theta Residual [%.5f]" % (
                                forward_iter + 1, forward_res_x.item(), forward_res_theta.item())
                              )

                        self.save_results(x_hat, theta_hat, x_gt, theta_gt, forward_iter)

            x_hat, theta_hat = x_pre, theta_pre

        return x_hat, theta_hat

    def save_results(self, x_pre, theta_pre, x_gt, theta_gt, forward_iter):
        save_path = get_save_path_from_config(self.config)
        
        x_pre = torch.clone(x_pre)
        theta_pre = torch.clone(theta_pre)
        
        x_pre[x_gt == 0] = 0
        theta_pre[theta_gt == 0] = 0

        x_pre = abs(x_pre)[0].detach().cpu().numpy()
        x_pre = (x_pre - np.amin(x_pre)) / (np.amax(x_pre) - np.amin(x_pre))
        x_pre = (x_pre * 255).astype(np.uint8)

        theta_pre = abs(theta_pre)[0, 5].detach().cpu().numpy()
        theta_pre = (theta_pre - np.amin(theta_pre)) / (np.amax(theta_pre) - np.amin(theta_pre))
        theta_pre = (theta_pre * 255).astype(np.uint8)

        imsave(os.path.join(save_path, f"estimated_image_iter{forward_iter + 1}.png"), x_pre)
        imsave(os.path.join(save_path, f"estimated_coil_sensitivity_map_iter{forward_iter + 1}.png"), theta_pre)
        
        if (forward_iter + 1) == 50:
            x_gt = torch.clone(x_gt)
            theta_gt = torch.clone(theta_gt)
            
            x_gt = abs(x_gt)[0].detach().cpu().numpy()
            x_gt = (x_gt - np.amin(x_gt)) / (np.amax(x_gt) - np.amin(x_gt))
            x_gt = (x_gt * 255).astype(np.uint8)

            theta_gt = abs(theta_gt)[0, 5].detach().cpu().numpy()
            theta_gt = (theta_gt - np.amin(theta_gt)) / (np.amax(theta_gt) - np.amin(theta_gt))
            theta_gt = (theta_gt * 255).astype(np.uint8)

            imsave(os.path.join(save_path, f"gt_image.png"), x_gt)
            imsave(os.path.join(save_path, f"gt_coil_sensitivity_map.png"), theta_gt)
            
    def step_helper(self, batch):
        x_input, theta_input, y, mask, x_gt, theta_gt = batch

        x_hat, theta_hat = self(x_input, theta_input, mask, y, x_gt, theta_gt)  # gt is used only for visualization.

        ssim_x, psnr_x = self.ssim_psnr_helper(x_hat=x_hat, x_gt=x_gt, data_range=1)

        theta_hat[:, :, torch.abs(x_gt[0]) == 0] = 0
        theta_gt[:, :, torch.abs(x_gt[0]) == 0] = 0
        mse_theta = self.mse_helper(theta_hat, theta_gt)

        return psnr_x, ssim_x, mse_theta, x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt

    def test_step(self, batch, batch_idx):
        psnr_x, ssim_x, mse_theta, x_input, x_hat, x_gt, theta_input, theta_hat, theta_gt = self.step_helper(batch)

        self.log(name='Image PSNR', value=psnr_x, prog_bar=True)
        self.log(name='Theta RMSE', value=mse_theta, prog_bar=True)
        self.log(name='Image SSIM', value=ssim_x, prog_bar=True)

        res = {
            'image_hat': x_hat,
            'image_gt': x_gt,
            'image_input': x_input,
        }

        return res

    def test_epoch_end(self, outputs) -> None:

        log_dict, img_dict = convert_pl_outputs(outputs)

        # save_path = get_save_path_from_config(self.config)

        # check_and_mkdir(save_path)
        # write_test(
        #     save_path=save_path,
        #     log_dict=log_dict,
        #     img_dict=img_dict
        # )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001)

        return optimizer

    @staticmethod
    def mse_helper(z_hat, z_gt):
        z_hat[abs(z_gt) == 0] = 0

        return torch.norm(z_hat - z_gt) / torch.norm(z_gt)

    @staticmethod
    def ssim_psnr_helper(x_hat, x_gt, data_range):
        if x_hat.dtype == torch.complex64:
            x_hat = torch.abs(x_hat)
            x_gt = torch.abs(x_gt)

        x_hat /= x_hat.max()
        x_gt /= x_gt.max()

        if x_hat.shape[-2] == int(2 * x_hat.shape[-1]):
            half_dim = x_hat.shape[-2] // 4

            x_hat = x_hat[..., half_dim:-half_dim, :]
            x_gt = x_gt[..., half_dim:-half_dim, :]

        x_hat[x_gt == 0] = 0

        if x_hat.dim() == 3:
            x_hat = x_hat.unsqueeze(1)
            x_gt = x_gt.unsqueeze(1)

        elif x_hat.dim() == 2:
            x_hat = x_hat.unsqueeze(0).unsqueeze(0)
            x_gt = x_gt.unsqueeze(0).unsqueeze(0)

        return structural_similarity_index_measure(x_hat, x_gt, data_range=data_range), peak_signal_noise_ratio(x_hat, x_gt, data_range=data_range)


class TestDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, item):

        with open('data/data.pt', 'rb') as f:
            x_hat, smps_hat, y, mask, x, smps = pickle.load(f)

        return x_hat, smps_hat, y, mask, x, smps


def run(config):

    trainer = get_trainer_from_config(config)

    tst_dataset = TestDataset()

    save_path = get_save_path_from_config(config)
    check_and_mkdir(save_path)

    print("\n====================================================================")
    print("Runing BC-PnP algorithm for jointly estimating coil sensitivity maps")
    print("====================================================================\n")

    model = BCPnP(
        net_x=get_module_from_config(config, type_='x', use_sigma_map=(config['method']['bcpnp']['warmup']['x_ckpt'] == 'g_denoise')),
        net_theta=get_module_from_config(config, type_='theta', use_sigma_map=(config['method']['bcpnp']['warmup']['theta_ckpt'] == 'g_denoise')),
        config=config
    )

    tst_dataloader = DataLoader(
        tst_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

    trainer.test(
        model=model,
        dataloaders=tst_dataloader,
        ckpt_path=None
    )
