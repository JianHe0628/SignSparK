import torch
import torch.nn as nn
import copy
import os
from multilingual_clip import pt_multilingual_clip
import transformers
from sentence_transformers import SentenceTransformer  
from .unet_model_large import TemporalUnetLarge

# Disable tokenizer parallelism to avoid fork warnings with DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Wilor_UNET(nn.Module):
    """
    Diffuser's style UNET
    """

    def __init__(self,
                 njoints=90,
                 nfeats=1,
                 latent_dim=512,
                 dim_mults=(1, 2, 4, 8),
                 attention=True,
                 dataset='wilor',
                 adagn=True,
                 zero=True,
                 arch='unet',
                 unet_out_mult=8,
                 keyframe_conditioned=True,
                 zero_keyframe_loss=False,
                 length_mask_conditioned=True,
                 text_conditioned=True,
                 text_mask_prob=0.1,
                 final_type=2,  # 1, 2, 3
                 text_enc_name="CLIP",
                 text_enc_dim=640,
                 text_enc_ver='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
                 ):
        super().__init__()

        self.njoints = njoints
        self.nfeats = nfeats
        self.dataset = dataset
        self.text_encoder_name = text_enc_name

        self.latent_dim = latent_dim
        self.dim_mults = dim_mults
        self.attention = attention
        self.input_feats = self.njoints * self.nfeats
        self.text_cond = text_conditioned
        self.cond_mask_prob = text_mask_prob

        self.text_enc_dim = text_enc_dim
        self.text_enc_ver = text_enc_ver

        self.keyframe_conditioned = keyframe_conditioned
        self.zero_keyframe_loss = zero_keyframe_loss
        self.length_mask_conditioned = length_mask_conditioned

        if self.keyframe_conditioned:
            added_channels = self.input_feats
        else:
            added_channels = 0
        if self.length_mask_conditioned:
            added_channels += self.input_feats
        # no need for the input process any
        # NOTE: just a mock, doing nothing
        # self.input_process = InputProcess(self.data_rep, self.input_feats)

        self.embed_timestep = TimestepEmbedder(self.latent_dim)  # Remove sequence_pos_encoder dependency

        print(f'Using UNET with: Input Dim {self.input_feats} | Latent Dim: {self.latent_dim} | Mults: {self.dim_mults} | Added channels: {added_channels}')
        print(f'AdaGN: {adagn} | Zero: {zero} | Attention: {attention}')
        self.unet = TemporalUnetLarge(input_dim=self.input_feats,
                                      cond_dim=self.latent_dim,
                                      dim=self.latent_dim,
                                      dim_mults=(1, 2, 4, 4),
                                      attention=self.attention,
                                      adagn=adagn,
                                      zero=zero,
                                      out_mult=unet_out_mult,
                                      added_input_channels=added_channels,
                                      final_type=final_type)

        if self.text_cond:
            self.embed_text = nn.Linear(self.text_enc_dim, self.latent_dim)
            if self.text_encoder_name == "CLIP":
                print(f'Text Conditioning Enabled. Loading CLIP {self.text_enc_ver}...')
                self.text_enc_model, self.text_enc_tokenizer = self.load_and_freeze_clip(self.text_enc_ver)
            elif self.text_encoder_name == "GEMMA":
                print(f'Text Conditioning Enabled. Loading Gemma {self.text_enc_ver}...')
                self.text_enc_model = SentenceTransformer(self.text_enc_ver)
                self.text_enc_model = self.freeze_params(self.text_enc_model)
            elif self.text_encoder_name == "QWEN":
                print(f'Text Conditioning Enabled. Loading QWEN {self.text_enc_ver}...')
                self.text_enc_model = SentenceTransformer(self.text_enc_ver, )
                self.text_enc_model = self.freeze_params(self.text_enc_model)
            else:
                raise NotImplementedError(f'Unknown text encoder {self.text_encoder_name}')

    def parameters_wo_clip(self):
        return [
            p for name, p in self.named_parameters()    
            if not name.startswith('clip_model.')
        ]

    def freeze_params(self, model):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def load_and_freeze_clip(self, clip_version):

        # Load Model & Tokenizer
        clip_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(clip_version, device='cpu')
        clip_tokenizer = transformers.AutoTokenizer.from_pretrained(clip_version)
        # Note: tokenizer doesn't have device param - it always returns CPU tensors

        return self.freeze_params(clip_model), clip_tokenizer

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            # bernoulli generates 0/1 based on the input probability given
            # we thus make mask of 1s then multiply by the probability (0.1)
            mask = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        # device = next(self.parameters()).device
        
        # Don't use clip_model.forward() - it doesn't move tokenizer outputs to device!
        # Instead, manually tokenize and move to correct device
        with torch.no_grad():
            if self.text_encoder_name == "CLIP":
                txt_tok = self.text_enc_tokenizer(raw_text, padding=True, return_tensors='pt')
                # Move to same device as CLIP model
                txt_tok = {key: val.to(self.text_enc_model.transformer.device) for key, val in txt_tok.items()}
                embs = self.text_enc_model.transformer(**txt_tok)[0]
                # Mean pooling with attention mask
                embs = (embs * txt_tok['attention_mask'].unsqueeze(-1)).sum(dim=1) / txt_tok['attention_mask'].sum(dim=1, keepdim=True)
                embeddings = self.text_enc_model.LinearTransformation(embs)
            elif self.text_encoder_name == "GEMMA":
                embeddings = self.text_enc_model.encode_document(raw_text, convert_to_tensor=True, show_progress_bar=False, batch_size=len(raw_text))
            elif self.text_encoder_name == "QWEN":
                embeddings = self.text_enc_model.encode(raw_text, convert_to_tensor=True, show_progress_bar=False, batch_size=len(raw_text))

        return embeddings

    def forward(self, x, timesteps, y=None, obs_x0=None, obs_mask=None):
        """
        Args:
            x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)
            y: dict of conditioning information
            obs_x0: [batch_size, njoints, nfeats, max_frames], observed keyframes
            obs_mask: [batch_size, njoints, nfeats, max_frames], mask for the observed keyframes

        Returns: [batch_size, njoints, nfeats, max_frames]
        """
        assert (obs_x0 is None) == (obs_mask is None), 'with spatial-conditioning, both obs_x0 and obs_mask must be provided'
        x = x.unsqueeze(2) if x.dim() == 3 else x
        obs_x0 = obs_x0.unsqueeze(2) if obs_x0.dim() == 3 else obs_x0
        obs_mask = obs_mask.unsqueeze(2) if obs_mask.dim() == 3 else obs_mask
        
        length_mask = copy.deepcopy(y['mask'])
        if length_mask is not None:
            length_mask = length_mask.to(x.device)
            length_mask = length_mask.unsqueeze(2) if length_mask.dim() == 3 else length_mask
            # Expand [B,1,1,T] to [B,J,1,T] so it matches the pose channels.
            if length_mask.size(1) == 1 and length_mask.size(2) == 1:
                length_mask = length_mask.expand(-1, obs_mask.size(1), -1, -1)

        if self.keyframe_conditioned:
            assert self.dataset in ['wilor', 'fullbody', 'bodyonly', 'face', 'hand', 'body', 'full'], f'Unknown dataset {self.dataset}'
            x = obs_x0 * obs_mask + x * (~obs_mask)
            if self.length_mask_conditioned and length_mask is not None:
                x = torch.cat([x, obs_mask, length_mask], dim=1)
            else:
                x = torch.cat([x, obs_mask], dim=1)
        else:
            print('You have not enabled keyframe conditioning!')
        
        return self.forward_core(x, timesteps, y)

    def forward_core(self, x, timesteps, y=None):
        """
        Args:
            x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
            timesteps: [batch_size] (int)

        Returns: [batch_size, njoints, nfeats, max_frames]
        """
        bs, njoints, nfeats, nframes = x.shape
        emb = self.embed_timestep(timesteps)  # [bs, d] - continuous time embedding

        force_mask = y.get('uncond', False) if y else False
        if self.text_cond and y:
            enc_text = self.encode_text(y['text']) # [bs, text_enc_dim] The sequential aspect of text is lost
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))

        # no need for positional embedding in the input becuase we use convolution.
        # [nframes, bs, d]
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints * nfeats)

        # multiplier of 16 for the unet temporal dim
        x = self.unet(x, cond=emb)  # , src_key_padding_mask=~maskseq)  # [nframes, bs, d]

        # [bs, njoints, nframes]
        x = x.permute(1, 2, 0)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        time_embed_dim = self.latent_dim * 4  # Flow convention: 4x expansion
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),  # SiLU activation for flow models
            nn.Linear(time_embed_dim, self.latent_dim),
        )

    def forward(self, timesteps):
        # Use continuous timestep embedding instead of positional encoding lookup
        from .utils.nn import timestep_embedding
        return self.time_embed(timestep_embedding(timesteps, self.latent_dim))