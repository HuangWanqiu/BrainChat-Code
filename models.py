
import argparse
import os
from typing import Callable, Optional, Sequence
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from timm.models.vision_transformer import Block


try:

    from transformers import (
        BeamSearchScorer,
        LogitsProcessorList,
        TopPLogitsWarper,
        TopKLogitsWarper,
        RepetitionPenaltyLogitsProcessor,
        MinLengthLogitsProcessor,
        MaxLengthCriteria,
        StoppingCriteriaList
    )
    GENERATION_TYPES = {
        "top_k": TopKLogitsWarper,
        "top_p": TopPLogitsWarper,
        "beam_search": "beam_search"
    }
    _has_transformers = True
except ImportError as e:
    GENERATION_TYPES = {
        "top_k": None,
        "top_p": None,
        "beam_search": "beam_search"
    }
    _has_transformers = False

class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """
    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, V = x.shape # batch, channel, voxels: torch.Size([16, 1, 4192])
        # assert V == self.num_voxels, \
        #     f"Input fmri length ({V}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous() # put embed_dim at the last dimension
        # self.proj(x): B, embed_dim (out_chans, length), V/patch_size (num_patches)
        return x
class fmri_encoder(nn.Module):
    def __init__(self, num_voxels=224, patch_size=16, embed_dim=1024, in_chans=1,
                 depth=24, num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, global_pool=True):
        super().__init__()
        self.patch_embed = PatchEmbed1D(num_voxels, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
    
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.embed_dim = embed_dim

        self.patch_size = patch_size
        self.num_patches = num_patches
        self.global_pool = global_pool
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.num_patches, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def split_tokens_embeds(self, x, pool_type: str = 'first'):
        if pool_type == 'first':
            tokens, embeds = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            tokens, embeds = x[:, -1], x[:, :-1]
        else:
            tokens = embeds = x
        return tokens, embeds
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x.mean(dim=1, keepdim=True)
        x = self.norm(x)

        return x  

    def forward(self, imgs):
        if imgs.ndim == 2:
            imgs = torch.unsqueeze(imgs, dim=0)  # N, n_seq, embed_dim
        latent = self.forward_encoder(imgs) # N, n_seq, embed_dim
        return latent # N, n_seq, embed_dim
    
    def load_checkpoint(self, state_dict):
        if self.global_pool:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k and 'norm' not in k)}
        else:
            state_dict = {k: v for k, v in state_dict.items() if ('mask_token' not in k)}
        ut.interpolate_pos_embed(self, state_dict)
            
        m, u = self.load_state_dict(state_dict, strict=False)
        print('missing keys:', u)
        print('unexpected keys:', m)
        return 
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        ''' add by wqhuang '''
        for param in self.parameters():
            param.requires_grad = False
        if unlocked_groups != 0:
            groups = [
                [
                    self.patch_embed,
                    self.pos_embed,
                ],
                *self.blocks[:-1],
                [
                    self.blocks[-1],
                    self.norm,
                ],
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])
class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int, # out dim
            context_dim: int, # in dim
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = nn.LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD
def create_model_and_transforms_coca():
    # Pseudocode, omitting the specific implementation of the create_model_and_transforms_coca
    # for specific implementation details, refer to https://github.com/mlfoundations/open_clip
    pass
class FCoCa(nn.Module):
    def __init__(
            self,
            metafile_fMRI,
            metafile_path_coca,
            num_voxels, 
            global_pool,
            model_name_coca="coca_ViT-L-14",
    ):
        super(FCoCa, self).__init__()
        ''' load pretrained fMRI encoder '''
        config = metafile_fMRI['config']
        # self.fMRI = create_model_fMRI_from_config(metafile_fMRI['config'], num_voxels, global_pool)
        self.fMRI = fmri_encoder(num_voxels=num_voxels, patch_size=config.patch_size, embed_dim=config.embed_dim, depth=config.depth, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio, global_pool=global_pool) 
        if metafile_fMRI['model']: # for Exp 5 and 10 (w/o fMRI encoder)
            self.fMRI.load_checkpoint(metafile_fMRI['model'])
        self.fmri_embed_dim = metafile_fMRI['config'].embed_dim

        ''' load trained coca'''
        self.coca, _, self.transform, coca_model_cfg = create_model_and_transforms_coca(
            model_name = model_name_coca,
            pretrained = metafile_path_coca,
            # device = device
        )
        self.coca_model_cfg = coca_model_cfg
        self.coca_embed_dim = coca_model_cfg['embed_dim']
        ''' project fMRI embedding (1024 dim) to coca decoder embedding (768) '''
        # method 1
        self.fMRI2coca_emb = nn.Linear(self.fmri_embed_dim, coca_model_cfg['embed_dim'], bias=True) # FIXME: refer coca vision 1024 -> 768
        self.fMRI2coca_latent = nn.Linear(self.fmri_embed_dim, coca_model_cfg['embed_dim'], bias=True) # FIXME: refer coca vision 1024 -> 768
        # method 2
        self.fMRI2coca = AttentionalPooler(
                    d_model = self.coca_embed_dim,
                    context_dim = self.fmri_embed_dim,
                    n_head=config.num_heads,
                    # n_queries=attn_pooler_queries,
                )
        self.ln_post = nn.LayerNorm(coca_model_cfg['embed_dim'])
    def _encode_fMRI_m1(self, fMRI, normalize: bool = True):
        x = self.fMRI(fMRI)

        fMRI_latent, fMRI_emb = self.fMRI.split_tokens_embeds(x, pool_type = 'first')
        fMRI_latent = F.normalize(fMRI_latent, dim=-1) if normalize else fMRI_latent
        fMRI_emb = self.fMRI2coca_emb(fMRI_emb)
        fMRI_latent = self.fMRI2coca_latent(fMRI_latent)
        return fMRI_latent, fMRI_emb
    def _encode_fMRI(self, fMRI, normalize: bool = True):
        x = self.fMRI(fMRI)
        x = self.fMRI2coca(x)
        x = self.ln_post(x)
        fMRI_latent, fMRI_emb = self.fMRI.split_tokens_embeds(x, pool_type = 'first')
        return fMRI_latent, fMRI_emb
    def forward_img_text(
        self,
        image,
        text: Optional[torch.Tensor] = None,
        image_latent: Optional[torch.Tensor] = None,
        image_embs: Optional[torch.Tensor] = None,
    ): # no fMRI
        if image_latent is None or image_embs is None:
            image_latent, image_embs = self.coca._encode_image(image)

        if text is None:
            return {"image_features": image_latent, "image_embs": image_embs}

        text_latent, token_embs = self.coca._encode_text(text)

        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1]:]

        logits = self.coca.text_decoder(image_embs, token_embs)
        out_dict = {
            "image_features": image_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.coca.logit_scale.exp()
        }
        if self.coca.logit_bias is not None:
            out_dict["logit_bias"] = self.coca.logit_bias
        return out_dict
        
    def forward_fMRI_text(
        self,
        fMRI,
        text: Optional[torch.Tensor] = None,
    ):
        fMRI_latent, fMRI_embs = self._encode_fMRI(fMRI)

        if text is None:
            return {"fMRI_features": fMRI_latent, "fMRI_embs": fMRI_embs}

        text_latent, token_embs = self.coca._encode_text(text)

        # TODO: add assertion to avoid bugs?
        labels = text[:, -token_embs.shape[1]:]

        logits = self.coca.text_decoder(fMRI_embs, token_embs)
        out_dict = {
            "fMRI_features": fMRI_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.coca.logit_scale.exp()
        }
        if self.coca.logit_bias is not None:
            out_dict["logit_bias"] = self.coca.logit_bias
        return out_dict

    def forward(
        self,
        fMRI,
        image,
        text, # (B/N, L=[0, 76])
        double_conds = False,
        is_training = True
    ):
        fMRI_latent, fMRI_embs = self._encode_fMRI(fMRI)
        image_latent, image_embs = self.coca._encode_image(image)


        text_latent, token_embs = self.coca._encode_text(text) # text_latent: (B/N, dim = 768); token_embs: (B/N, L = [0, 76], dim = 768)

        # TODO: add assertion to avoid bugs?
        # labels = text[:, -token_embs.shape[1]:]
        labels = text[:, 1:]
        if is_training:
            token_embs = token_embs[:, :-1]

        if double_conds:
            cond_embs = self.get_condition(fMRI_embs, image_embs)
        else:
            cond_embs = fMRI_embs
            
        logits = self.coca.text_decoder(cond_embs, token_embs) # logits: (B/N, L = [0, 76], dim = 768)
        out_dict = {
            "image_features": image_latent,
            "fMRI_features": fMRI_latent,
            "text_features": text_latent,
            "logits": logits,
            "labels": labels,
            "logit_scale": self.coca.logit_scale.exp()
        }
        if self.coca.logit_bias is not None:
            out_dict["logit_bias"] = self.coca.logit_bias
        return out_dict
    def forward_vqa(self,
        fMRI,
        image,
        question, # (B/N, L=[0, 76])
        double_conds = False,
        is_ans_gen = False # for vqa generating answer loss
    ):
        fMRI_latent, fMRI_embs = self._encode_fMRI(fMRI)
        image_latent, image_embs = self.coca._encode_image(image)
        text_latent, token_embs = self.coca._encode_text(question) # text_latent: (B/N, dim = 768); token_embs: (B/N, L = [0, 76], dim = 768)
        # labels = text[:, -token_embs.shape[1]:]
        labels = None
        if is_ans_gen:
            labels = question[:, 1:]
            token_embs = token_embs[:, :-1]

        if is_ans_gen: # for vqa generating answer loss
            token_embs = token_embs[:, :-1]

        if double_conds:
            cond_embs = self.get_condition(fMRI_embs, image_embs)
        else:
            cond_embs = fMRI_embs
            
        logits, embs_final = self.coca.text_decoder(cond_embs, token_embs, return_embs_final = True) # logits: (B/N, L = [0, 76], dim = 768) # embs_final: (B/N, dim = 768)
        # embs_final = embs_final[:, -1] # pooler: get last
        out_dict = {
            "embs_final": embs_final,
            "labels": labels,
            "logits": logits,
            "image_features": image_latent,
            "fMRI_features": fMRI_latent,
            "text_features": text_latent,
        }
        return out_dict
    def generate(
        self,
        image,
        fMRI,
        text=None,
        seq_len=30,
        max_seq_len=77,
        temperature=1.,
        generation_type="top_p", # beam_search
        top_p=0.1,  # keep tokens in the 1 - top_p quantile
        top_k=1,  # keeps the top_k most probable tokens
        pad_token_id=None,
        eos_token_id=None,
        sot_token_id=None,
        num_beams=6,
        num_beam_groups=3,
        min_seq_len=5,
        stopping_criteria=None,
        repetition_penalty=1.0,
        fixed_output_length=False, # if True output.shape == (batch_size, seq_len)
        double_conds = False,
    ):
        # taking many ideas and components from HuggingFace GenerationMixin
        # https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
        assert _has_transformers, "Please install transformers for generate functionality. `pip install transformers`."
        assert seq_len > min_seq_len, "seq_len must be larger than min_seq_len"

        with torch.no_grad():
            sot_token_id = 49406 if sot_token_id is None else sot_token_id
            eos_token_id = 49407 if eos_token_id is None else eos_token_id
            pad_token_id = self.coca.pad_id if pad_token_id is None else pad_token_id
            logit_processor = LogitsProcessorList(
                [
                    MinLengthLogitsProcessor(min_seq_len, eos_token_id),
                    RepetitionPenaltyLogitsProcessor(repetition_penalty),
                ]
            )

            if stopping_criteria is None:
                stopping_criteria = [MaxLengthCriteria(max_length=seq_len)]

            stopping_criteria = StoppingCriteriaList(
                stopping_criteria
            )

            device = image.device

            if generation_type == "beam_search": # FIXME: finish beam_search
                output = self._generate_beamsearch(
                    image_inputs=image,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                    sot_token_id=sot_token_id,
                    num_beams=num_beams,
                    num_beam_groups=num_beam_groups,
                    min_seq_len=min_seq_len,
                    stopping_criteria=stopping_criteria,
                    logit_processor=logit_processor,
                )
                if fixed_output_length and output.shape[1] < seq_len:
                    return torch.cat(
                        (output, torch.ones(output.shape[0], seq_len-output.shape[1], device=device, dtype=output.dtype) * self.pad_id),
                        dim=1
                    )
                return output

            elif generation_type == "top_p":
                logit_warper = GENERATION_TYPES[generation_type](top_p)
            elif generation_type == "top_k":
                logit_warper = GENERATION_TYPES[generation_type](top_k)
            else:
                raise ValueError(
                    f"generation_type has to be one of "
                    f"{'| ' + ' | '.join(list(GENERATION_TYPES.keys())) + ' |'}."
                )

            # image_latent, image_embs = self.coca._encode_image(image) # image_latent: (B, 768); image_embs: (B, 255, 768);
            # fMRI_latent, fMRI_embs = self._encode_fMRI(fMRI)

            if text is None:
                text = torch.ones((image.shape[0], 1), device=device, dtype=torch.long) * sot_token_id # (B, 1)

            was_training = self.training
            num_dims = len(text.shape)

            if num_dims == 1:
                text = text[None, :]

            cur_len = text.shape[1]
            self.eval()
            out = text

            while True:
                x = out[:, -max_seq_len:]
                cur_len = x.shape[1]

                logits = self(fMRI, image, x, double_conds = double_conds, is_training = False)["logits"][:, -1] # why -1? # -1: see all x; 0: only see sot_token_id of x
                # logits = self(image, x, image_latent=image_latent, image_embs=image_embs)["logits"][:, -1] 

                mask = (out[:, -1] == eos_token_id) | (out[:, -1] == pad_token_id)
                sample = torch.ones((out.shape[0], 1), device=device, dtype=torch.long) * pad_token_id # (batch_size, 1)

                if mask.all():
                    if not fixed_output_length:
                        break
                else:
                    logits = logits[~mask, :]
                    filtered_logits = logit_processor(x[~mask, :], logits)
                    filtered_logits = logit_warper(x[~mask, :], filtered_logits)
                    probs = F.softmax(filtered_logits / temperature, dim=-1)

                    if (cur_len + 1 == seq_len):
                        sample[~mask, :] = torch.ones((sum(~mask), 1), device=device, dtype=torch.long) * eos_token_id
                    else:
                        sample[~mask, :] = torch.multinomial(probs, 1)

                out = torch.cat((out, sample), dim=-1)

                cur_len += 1

                if stopping_criteria(out, None):
                    break

            if num_dims == 1:
                out = out.squeeze(0)

            self.train(was_training)
            return out


    # FIXME: to revise get_condition()
    def get_condition(self, fMRI_embs, image_embs):
        return image_embs

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        ''' add by wqhuang '''
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.coca.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        ''' add by wqhuang '''
        self.coca.text.lock(unlocked_layers, freeze_layer_norm)

    def lock_fMRI_tower(self, unlocked_groups: int = 0, freeze_layer_norm: bool = True):
        ''' add by wqhuang '''
        self.fMRI.lock(unlocked_groups, freeze_layer_norm)

    def lock_coca(self):
        for param in self.coca.parameters():
            param.requires_grad = False

