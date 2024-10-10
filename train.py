
import torch

from tqdm import tqdm
import torchvision.transforms as transforms
import math
def train_one_epoch_fcoca(model, dataloader, loss, epoch, optimizer, scaler, scheduler, args):

    model.train()

    tqdm_data_loader = tqdm(enumerate(dataloader), total=len(dataloader), desc=f'[Epoch {epoch}]')
    for i, batch in tqdm_data_loader:
        scheduler(epoch * len(dataloader) + i)
        optimizer.zero_grad()
        images, fMRI, cap_tokens, img_ids, img_cn_tokens = batch

        to_pil = transforms.ToPILImage()
        images = torch.stack([model.transform(to_pil(image)) for image in images], dim=0)
        
        # mixup fMRI voxel
        mixco_info = None
        if args.mixco:
            # print("Mixco fMRI ...")
            fMRI, mixco_info = mixco(fMRI)

        model_out = model.forward(fMRI, images, cap_tokens)
        logit_scale = model_out["logit_scale"]
        losses = loss(**model_out, output_dict=True, mixco_info = mixco_info)
        total_loss = sum(losses.values())
        losses["loss"] = total_loss

        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.coca.logit_scale.clamp_(0, math.log(100))

       
def mixco(voxels, beta=0.15, s_thresh=0.5):
    perm = torch.randperm(voxels.shape[0]).to(voxels.device)
    voxels_shuffle = voxels[perm].to(voxels.device,dtype=voxels.dtype)
    
    betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(voxels.device,dtype=voxels.dtype)
    select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device) 
    betas_shape = [-1] + [1]*(len(voxels.shape)-1) # [-1, 1, ..., 1]

    voxels[select] = voxels[select] * betas[select].reshape(*betas_shape) + \
        voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    betas[~select] = 1 
    mixco_info = {
        "perm": perm, # torch.Size([B]) # tensor([ 2,  5, ..., 28])
        "betas": betas, # torch.Size([B]) # tensor([1.0000e+00, 1.0000e+00, 9.9152e-01, ..., 1.0000e+00])
        "select": select, # torch.Size([B]) # tensor([False, False,  True, ..., False])
    }
    return voxels, mixco_info