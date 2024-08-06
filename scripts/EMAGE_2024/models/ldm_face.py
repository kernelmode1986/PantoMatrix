import copy
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from .utils.layer import BasicBlock
from .motion_encoder import * 

from .diffusion.mdm import MDM
from .diffusion import gaussian_diffusion as gd
from .diffusion.respace import SpacedDiffusion, space_timesteps
from .diffusion.resample import create_named_schedule_sampler

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])


def create_model_and_diffusion(args):
    model = MDM(**get_model_args(args))
    diffusion = create_gaussian_diffusion(args)
    return model, diffusion


def get_model_args(args):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'

    # SMPL defaults
    data_rep = 'hml_vec'
    nfeats = 1


    return {'modeltype': args.modeltype,'pose_dims': args.pose_dims, 'nfeats': nfeats, 
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': args.cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version}


def create_gaussian_diffusion(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
    )

class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
        else:
            wav_data = wav_data.transpose(1, 2)
        out = self.feat_extractor(wav_data)
        return out.transpose(1, 2)

    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
    def forward(self, inputs):
        out = self.mlp(inputs)
        return out


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=60): 
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class ldm_transformer(nn.Module):
    def __init__(self, args):
        super(ldm_transformer, self).__init__()
        self.args = args   
        with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder_face = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=args.t_fix_pre)
        self.text_encoder_face = nn.Linear(300, args.audio_f) 
        self.text_encoder_face = nn.Linear(300, args.audio_f) 
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=args.t_fix_pre)
        self.text_encoder_body = nn.Linear(300, args.audio_f) 
        self.text_encoder_body = nn.Linear(300, args.audio_f) 

        self.audio_pre_encoder_face = WavEncoder(args.audio_f, audio_in=1)
        self.audio_pre_encoder_body = WavEncoder(args.audio_f, audio_in=1)
        
        self.at_attn_face = nn.Linear(args.audio_f*2, args.audio_f*2)
        self.at_attn_body = nn.Linear(args.audio_f*2, args.audio_f*2)
        
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = args.motion_f
        args_top.vae_test_dim = args.pose_dims+3+4
        self.motion_encoder = VQEncoderV6(args_top) # masked motion to latent bs t 333 to bs t 256
        
        # face decoder 
        self.feature2face = nn.Linear(args.audio_f*2, args.hidden_size)
        self.face2latent = nn.Linear(args.hidden_size, args.vae_codebook_size)
        self.transformer_de_layer = nn.TransformerDecoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.face_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=4)
        self.position_embeddings = PeriodicPositionalEncoding(self.args.hidden_size, period=self.args.pose_length, max_seq_len=self.args.pose_length)
        
        # motion decoder
        self.transformer_en_layer = nn.TransformerEncoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        self.audio_feature2motion = nn.Linear(args.audio_f, args.hidden_size)
        self.feature2motion = nn.Linear(args.motion_f, args.hidden_size)

        self.bodyhints_face = MLP(args.motion_f, args.hidden_size, args.motion_f)
        self.bodyhints_body = MLP(args.motion_f, args.hidden_size, args.motion_f)
        self.motion2latent_upper = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_hands = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_lower = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.wordhints_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=8)
        
        self.upper_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.hands_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.lower_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)

        self.face_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.upper_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.hands_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.lower_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)

        self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, self.args.pose_dims+3+4))
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self._reset_parameters()

        self.spearker_encoder_body = nn.Embedding(25, args.hidden_size)
        self.spearker_encoder_face = nn.Embedding(25, args.hidden_size)
        
        model, diffusion = create_model_and_diffusion(args)
        self.diffusion = diffusion
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)    
        self.model = model
        

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.args.hidden_size ** -0.5)
    
    def forward(self, in_audio=None, in_word=None, mode=None, in_motion=None, use_attentions=True):
        bs, t, c = in_motion.shape
        if in_word:
            in_word_face = self.text_pre_encoder_face(in_word)
            in_word_face = self.text_encoder_face(in_word_face)
            in_word_body = self.text_pre_encoder_body(in_word)
            in_word_body = self.text_encoder_body(in_word_body)
        
        in_audio_face = self.audio_pre_encoder_face(in_audio)
        in_audio_body = self.audio_pre_encoder_body(in_audio)
        if in_audio_face.shape[1] != in_motion.shape[1]:
            diff_length = in_motion.shape[1]- in_audio_face.shape[1]
            if diff_length < 0:
                in_audio_face = in_audio_face[:, :diff_length, :]
                in_audio_body = in_audio_body[:, :diff_length, :]
            else:
                in_audio_face = torch.cat((in_audio_face, in_audio_face[:,-diff_length:]),1)
                in_audio_body = torch.cat((in_audio_body, in_audio_body[:,-diff_length:]),1)

        # if use_attentions:           
        #     alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
        #     alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2)
        #     alpha_at_face = alpha_at_face.softmax(dim=-1)
        #     fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]
        #     alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
        #     alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
        #     alpha_at_body = alpha_at_body.softmax(dim=-1)
        #     fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        # else:
        #     fusion_face = in_word_face + in_audio_face
        #     fusion_body = in_word_body + in_audio_body
        
        in_audio_face = in_audio_face.mean(dim = 1)
        
        
        
        if mode == 'train': 
            step, weights = self.schedule_sampler.sample(bs, 0)
            in_motion = in_motion.permute(0,2,1).unsqueeze(2)   
            latent_face_rec = self.diffusion.forward(self.model, in_motion, step, in_audio_face)
            latent_face_rec = latent_face_rec.squeeze(2).permute(0,2,1)
            return {
                "latent_face_rec":latent_face_rec
                }
            
        elif mode == 'test':        
            sample_fn = self.diffusion.p_sample_loop
            model_kwargs = {}
            model_kwargs['emb_cond'] = in_audio_face

            sample = sample_fn(
                self.model,
                shape=(bs, self.args.latent_dim, 1, t),
                clip_denoised=False,
                model_kwargs=model_kwargs,
                skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )
            sample = sample.squeeze(2).permute(0,2,1)
            
            return {"latent_face_rec": sample}
        
        else:
            pass
        
        