import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from utils.metrics import PSNR
from utils.postprocessing_functions import SimplePostProcess
psnr_fn = PSNR(boundary_ignore=40)

import models

seed_everything(13)
post_process = SimplePostProcess(return_np=True)


class BurstM(pl.LightningModule):
    def __init__(self, input_channels=4):
        super(BurstM, self).__init__()        
        
        self.train_loss = nn.L1Loss()
        self.train_loss2 = nn.MSELoss()
        self.valid_psnr = PSNR(boundary_ignore=40)
        self.input_channels = input_channels

        # Initialize model based on input channels
        if input_channels == 1:
            # For grayscale: use modified BurstM for single-channel input
            self.burstm_model = models.BurstM.Neural_Warping_Grayscale().cuda()
        else:
            # For RAW: use default 4-channel BurstM
            self.burstm_model = models.BurstM.Neural_Warping().cuda()
        
    
    def forward(self, burst, scale=4, target_size=(192,192)):
        
        burst = burst[0]  # Extract from list wrapper: [B, num_frames, C, H, W]
        
        # Debug: print burst shape
        print(f"DEBUG forward: burst shape after unwrap = {burst.shape}")
        
        # Handle batch dimension - process each sample in batch
        batch_size = burst.shape[0]
        
        if batch_size == 1:
            # Single sample: [1, num_frames, C, H, W] -> [num_frames, C, H, W]
            burst = burst.squeeze(0)
            print(f"DEBUG forward: burst shape after squeeze = {burst.shape}")
            print(f"DEBUG forward: burst[0] shape = {burst[0].shape}")
            print(f"DEBUG forward: burst[1:] shape = {burst[1:].shape}")
            
            burst_ref = burst[0].unsqueeze(0).clone()  # [1, C, H, W]
            burst_src = burst[1:]  # [num_frames-1, C, H, W]
            
            print(f"DEBUG forward: burst_ref shape = {burst_ref.shape}")
            print(f"DEBUG forward: burst_src shape = {burst_src.shape}")
            
            burst_feat, ref, EstLrImg = self.burstm_model(burst_ref, burst_src, scale, target_size)
        else:
            # Multiple samples: process individually and stack results
            burst_feat_list = []
            ref_list = []
            EstLrImg_list = []
            
            for i in range(batch_size):
                burst_i = burst[i]  # [num_frames, C, H, W]
                print(f"DEBUG forward batch {i}: burst_i shape = {burst_i.shape}")
                
                burst_ref_i = burst_i[0].unsqueeze(0).clone()  # [1, C, H, W]
                burst_src_i = burst_i[1:]  # [num_frames-1, C, H, W]
                
                print(f"DEBUG forward batch {i}: burst_ref_i shape = {burst_ref_i.shape}")
                print(f"DEBUG forward batch {i}: burst_src_i shape = {burst_src_i.shape}")
                
                burst_feat_i, ref_i, EstLrImg_i = self.burstm_model(burst_ref_i, burst_src_i, scale, target_size)
                
                burst_feat_list.append(burst_feat_i)
                ref_list.append(ref_i)
                EstLrImg_list.append(EstLrImg_i)
            
            # Stack results: [B, C, H, W]
            burst_feat = torch.cat(burst_feat_list, dim=0)
            ref = torch.cat(ref_list, dim=0)
            EstLrImg = torch.cat(EstLrImg_list, dim=0)
        
        return burst_feat, ref, EstLrImg
    
    def training_step(self, train_batch, batch_idx):
        x, y, flow_vectors, meta_info, downsample_factor, target_size = train_batch
        
        # Handle downsample_factor - use first element if batched
        scale = downsample_factor[0].item() if downsample_factor.dim() > 0 else downsample_factor.item()
        
        pred, ref, EstLrImg = self.forward(x, scale, target_size)
        pred = pred.clamp(0.0, 1.0)
        loss = self.train_loss(pred, y) + self.train_loss2(EstLrImg, ref)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, flow_vectors, meta_info, downsample_factor, target_size = val_batch
        
        # Handle downsample_factor - use first element if batched
        scale = downsample_factor[0].item() if downsample_factor.dim() > 0 else downsample_factor.item()
        
        pred, ref, EstLrImg = self.forward(x, scale, target_size)
        pred = pred.clamp(0.0, 1.0)
        PSNR = self.valid_psnr(pred, y)
        
        return PSNR


    def validation_epoch_end(self, outs):
        # outs is a list of whatever you returned in `validation_step`
        PSNR = torch.stack(outs).mean()
        self.log('val_psnr', PSNR, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):  
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300, eta_min=1e-6)            
        # return [optimizer], [lr_scheduler]
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',  # or 'step' if you want to update per batch
                'frequency': 1
            }
        }


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)