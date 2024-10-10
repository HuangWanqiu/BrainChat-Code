import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class FCoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            fi_contrastive_loss_weight,
            ft_contrastive_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.fi_contrastive_loss_weight = fi_contrastive_loss_weight
        self.ft_contrastive_loss_weight = ft_contrastive_loss_weight
        # caption loss
        weight_ce = torch.ones(49408).to(f"cuda:{rank}") if world_size else torch.ones(49408)
        # ignore_index_ls = [pad_id, 49406, 49407]
        ignore_index_ls = [pad_id]
        for c_ind in ignore_index_ls:
            weight_ce[c_ind] = 0
        self.caption_loss = nn.CrossEntropyLoss(weight=weight_ce)
        # self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def _get_loss(self, fMRI_features, x_features, logits, labels, logit_scale, mixco_info = None):
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(fMRI_features, x_features, logit_scale, mixco_info = mixco_info)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1), # torch.Size([1, 49408, 76]) # (N, word_size, L)
            labels, # torch.Size([1, 76]) # (N, L) # tensor([[49406,   320,  3638,  6716,  4730,   593,   320,  4818,  6716,   269,   49407,   0, ..., 0]], device='cuda:0')
        )
        caption_loss = caption_loss * self.caption_loss_weight

        return clip_loss, caption_loss

    def forward(self, image_features, fMRI_features, text_features, logits, labels, logit_scale, output_dict=False, mixco_info = None):
        
        losses = {}

        clip_loss_fi, caption_loss = self._get_loss(fMRI_features, image_features, logits, labels, logit_scale, mixco_info = mixco_info)
        clip_loss_ft, _ = self._get_loss(fMRI_features, text_features, logits, labels, logit_scale, mixco_info = mixco_info)
        if self.fi_contrastive_loss_weight or self.ft_contrastive_loss_weight:
            clip_loss_fi = clip_loss_fi * self.fi_contrastive_loss_weight
            clip_loss_ft = clip_loss_ft  * self.ft_contrastive_loss_weight
            losses["con (fi)"] = clip_loss_fi
            losses["con (ft)"] = clip_loss_ft

        if self.caption_loss_weight:
            losses["cap"] = caption_loss

        if output_dict:
            return losses
        else:
            return list(losses.values())
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # NOTE: When using `accelerate`, it is important to pay attention to whether modifications are needed here.
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T # image_features, text_features: (B/N, 768)
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text
    
    def mixco_nce(self, brain_clip, brain_clip_T, temp = 0.1, mixco_info = None, bidirectional=True):
        # brain_clip = (preds @ targs.T)/temp
        perm, betas, select = mixco_info["perm"], mixco_info["betas"], mixco_info["select"]
        perm, betas, select = map(lambda x: gather_feature(x, self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod), [perm, betas, select])

        probs = torch.diag(betas) # probs: torch.Size([32, 32]) # betas: torch.Size([32])
        # probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas
        probs[torch.arange(brain_clip.shape[0]).to(brain_clip.device), perm] = 1 - betas
        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            # loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss2 = -(brain_clip_T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2)/2
        return loss
    
    def forward(self, image_features, text_features, logit_scale, output_dict=False, mixco_info = None):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale) # (N, N)
        if mixco_info:
            total_loss = self.mixco_nce(logits_per_image, logits_per_text, mixco_info = mixco_info)
        else:
            labels = self.get_ground_truth(device, logits_per_image.shape[0]) # labels: range(0, N) # tensor([0, 1, 2], device='cuda:0') 
            total_loss = (
                F.cross_entropy(logits_per_image, labels) + # [100, 200]
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features.contiguous())
            dist.all_gather(gathered_text_features, text_features.contiguous())
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features
    
def gather_feature(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    if gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered_features, features.contiguous())
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_features[rank] = features
        all_features = torch.cat(gathered_features, dim=0)
    return all_features