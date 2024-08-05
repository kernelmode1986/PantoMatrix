import train
import os
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
import smplx
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa

from flame_pytorch import FLAME
import argparse 
import utils.media
import utils.fast_render

class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.joints = self.train_data.joints
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker(["ver_lvd", "ver_mse", "fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face",  "face", "face_vel", "face_acc", "ver", "ver_vel", "ver_acc"], [True,True,False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False,False,False])
        
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106
        self.vq_model_face = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        # print(self.vq_model_face)
        other_tools.load_checkpoints(self.vq_model_face, self.args.data_path_1 +  "pretrained_vq/last_790_face_v2.bin", args.e_name)
        self.args.vae_test_dim = 78
        self.vq_model_upper = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_upper, self.args.data_path_1 +  "pretrained_vq/upper_vertex_1layer_710.bin", args.e_name)
        self.args.vae_test_dim = 180
        self.vq_model_hands = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hands, self.args.data_path_1 +  "pretrained_vq/hands_vertex_1layer_710.bin", args.e_name)
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.vq_model_lower = getattr(vq_model_module, "VQVAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_lower, self.args.data_path_1 +  "pretrained_vq/lower_foot_600.bin", args.e_name)
        self.args.vae_test_dim = 61
        self.args.vae_layer = 4
        self.global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.global_motion, self.args.data_path_1 +  "pretrained_vq/last_1700_foot.bin", args.e_name)
        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240

        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
        self.global_motion.eval()

        self.cls_loss = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)

        self.flame_dict = {
                    'flame_model_path': self.args.data_path_1+"flame_models/generic_model.pkl",
                    'static_landmark_embedding_path': self.args.data_path_1+"flame_models/flame_static_embedding.pkl",
                    'dynamic_landmark_embedding_path': self.args.data_path_1+"flame_models/flame_static_embedding.pkl",
                    #flame_config.shape_params = 
                    'shape_params': 100,
                    'expression_params': 100,
                    'batch_size': 1, #change when use 
                    'use_face_contour': False,
                    'use_3D_translation': True}
        
      
    
    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_contact = tar_pose_raw[:, :, 165:169].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)
        tar_id = dict_data["id"].to(self.rank).long()
        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)

        # tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
        tar4dis = torch.cat([tar_pose_jaw, tar_pose_upper, tar_pose_hands, tar_pose_leg], dim=2)

        tar_index_value_face_top = self.vq_model_face.map2index(tar_pose_face) # bs*n/4
        tar_index_value_upper_top = self.vq_model_upper.map2index(tar_pose_upper) # bs*n/4
        tar_index_value_hands_top = self.vq_model_hands.map2index(tar_pose_hands) # bs*n/4
        tar_index_value_lower_top = self.vq_model_lower.map2index(tar_pose_lower) # bs*n/4

        latent_face_encoder = self.vq_model_face.encoder(tar_pose_face) # bs*n/4      
        latent_face_top = self.vq_model_face.map2latent(tar_pose_face) # bs*n/4
        latent_upper_top = self.vq_model_upper.map2latent(tar_pose_upper) # bs*n/4
        latent_hands_top = self.vq_model_hands.map2latent(tar_pose_hands) # bs*n/4
        latent_lower_top = self.vq_model_lower.map2latent(tar_pose_lower) # bs*n/4
        
        latent_in = torch.cat([latent_upper_top, latent_hands_top, latent_lower_top], dim=2)
        
        index_in = torch.stack([tar_index_value_upper_top, tar_index_value_hands_top, tar_index_value_lower_top], dim=-1).long()
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        # print(tar_index_value_upper_top.shape, index_in.shape)
        return {
            "tar_pose_jaw": tar_pose_jaw,
            "tar_pose_face": tar_pose_face,
            "tar_pose_upper": tar_pose_upper,
            "tar_pose_lower": tar_pose_lower,
            "tar_pose_hands": tar_pose_hands,
            'tar_pose_leg': tar_pose_leg,
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "tar4dis": tar4dis,
            "tar_index_value_face_top": tar_index_value_face_top,
            "tar_index_value_upper_top": tar_index_value_upper_top,
            "tar_index_value_hands_top": tar_index_value_hands_top,
            "tar_index_value_lower_top": tar_index_value_lower_top,
            "latent_face_encoder": latent_face_encoder,
            "latent_face_top": latent_face_top,
            "latent_upper_top": latent_upper_top,
            "latent_hands_top": latent_hands_top,
            "latent_lower_top": latent_lower_top,
            "latent_in":  latent_in,
            "index_in": index_in,
            "tar_id": tar_id,
            "latent_all": latent_all,
            "tar_pose_6d": tar_pose_6d,
            "tar_contact": tar_contact,
        }
    
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        # ------ full generatation task ------ #
        mask_val = torch.ones(bs, n, self.args.pose_dims+3+4).float().cuda()
        mask_val[:, :self.args.pre_frames, :] = 0.0
        
        net_out_val  = self.model(
            loaded_data['in_audio'], loaded_data['in_word'], mask=mask_val, mode='train',
            in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_face_top'],#no codebook 1.latent_face_encoder
            use_attentions = True)
        
        
        g_loss_final = 0
        
        tar_face = loaded_data['tar_pose_face']
        rec_face = self.vq_model_face.decoder(net_out_val['latent_face_rec']) #[32,64,256]
        
        joints = 1   #如果是pose，修改joints，使用上面的latent_in即可 
        j = 1        
        rec_pose = rc.rotation_6d_to_matrix(rec_face[:, :, :joints*6].reshape(bs, n, joints, 6))
        tar_pose = rc.rotation_6d_to_matrix(tar_face[:, :, :joints*6].reshape(bs, n, joints, 6))    
        rec_exps = rec_face[:, :, joints*6:]            
        tar_exps = tar_face[:, :, joints*6:]
        
        # jaw open 6d rec loss
        loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight
        self.tracker.update_meter("rec", "train", loss_rec.item())
        g_loss_final += loss_rec
        # jaw open 6d vel and acc loss
        velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * self.args.rec_weight
        acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * self.args.rec_weight
        self.tracker.update_meter("vel", "train", velocity_loss.item())
        self.tracker.update_meter("acc", "train", acceleration_loss.item())
        g_loss_final += velocity_loss 
        g_loss_final += acceleration_loss
        

        loss_face = self.reclatent_loss(rec_exps, tar_exps) * self.args.rec_weight
        self.tracker.update_meter("face", "train", loss_face.item())
        g_loss_final += loss_face
        # face parameter l1 vel and acc loss
        face_velocity_loss =  self.vel_loss(rec_exps[:, 1:] - rec_exps[:, :-1], tar_exps[:, 1:] - tar_exps[:, :-1]) * self.args.rec_weight
        face_acceleration_loss =  self.vel_loss(rec_exps[:, 2:] + rec_exps[:, :-2] - 2 * rec_exps[:, 1:-1], tar_exps[:, 2:] + tar_exps[:, :-2] - 2 * tar_exps[:, 1:-1]) * self.args.rec_weight
        self.tracker.update_meter("face_vel", "train", face_velocity_loss.item())
        self.tracker.update_meter("face_acc", "train", face_acceleration_loss.item())
        g_loss_final += face_velocity_loss
        g_loss_final += face_acceleration_loss
        
            # vertices loss
        if self.args.rec_ver_weight > 0:
            tar_trans = loaded_data['tar_trans']
            tar_beta = loaded_data['tar_beta']
            tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
            rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
            # vertices_rec = self.smplx(
            #     betas=tar_beta.reshape(bs*n, 300), 
            #     transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
            #     expression=rec_exps.reshape(bs*n, 100), 
            #     jaw_pose=rec_pose, 
            #     global_orient=torch.zeros(bs*n, 3).cuda(), 
            #     body_pose=torch.zeros(bs*n, 21*3).cuda(), 
            #     left_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
            #     right_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
            #     return_verts=True,
            #     # return_joints=True,
            #     leye_pose=torch.zeros(bs*n, 3).cuda(), 
            #     reye_pose=torch.zeros(bs*n, 3).cuda(),
            # )
            # vertices_tar = self.smplx(
            #     betas=tar_beta.reshape(bs*n, 300), 
            #     transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
            #     expression=tar_exps.reshape(bs*n, 100), 
            #     jaw_pose=tar_pose, 
            #     global_orient=torch.zeros(bs*n, 3).cuda(), 
            #     body_pose=torch.zeros(bs*n, 21*3).cuda(), 
            #     left_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
            #     right_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
            #     return_verts=True,
            #     # return_joints=True,
            #     leye_pose=torch.zeros(bs*n, 3).cuda(), 
            #     reye_pose=torch.zeros(bs*n, 3).cuda(),
            # )  


            self.flame_dict['batch_size'] = bs*n
            self.flame = FLAME(argparse.Namespace(**self.flame_dict)).to(self.rank).eval()
            shape_params = torch.zeros(bs*n, 100).cuda()
            rec_pose_params = torch.zeros(bs*n, 6).cuda()
            rec_pose_params[:, 3:] = rec_pose
            
            tar_pose_params = torch.zeros(bs*n, 6).cuda()
            tar_pose_params[:, 3:] = tar_pose
            
            rec_expression_params = rec_exps.reshape(bs*n, 100)
            rec_flame_vertices, rec_flame_landmarks = self.flame(
                shape_params, rec_expression_params, rec_pose_params
            )
            tar_expression_params = tar_exps.reshape(bs*n, 100)
            tar_flame_vertices, tar_flame_landmarks = self.flame(
                shape_params, tar_expression_params, tar_pose_params
            )
            vertices_rec = {'vertices':rec_flame_vertices, 'landmarks':rec_flame_landmarks}
            vertices_tar = {'vertices':tar_flame_vertices, 'landmarks':tar_flame_landmarks}



            vectices_loss = self.reclatent_loss(vertices_rec['vertices'], vertices_tar['vertices'])
            self.tracker.update_meter("ver", "train", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
            g_loss_final += vectices_loss*self.args.rec_weight*self.args.rec_ver_weight
            # vertices vel and acc loss
            # vert_velocity_loss =  self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1]) * self.args.rec_weight * self.args.rec_ver_weight
            # vert_acceleration_loss =  self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1]) * self.args.rec_weight * self.args.rec_ver_weight
            # self.tracker.update_meter("ver_vel", "train", vert_velocity_loss.item())
            # self.tracker.update_meter("ver_acc", "train", vert_acceleration_loss.item())
            # g_loss_final += vert_velocity_loss
            # g_loss_final += vert_acceleration_loss

        if mode == 'train':
            return g_loss_final
        elif mode == 'val':
            return {
                'rec_pose': rec_pose,
                # rec_trans': rec_pose_trans,
                'tar_pose': loaded_data["tar_pose_6d"],
            }
        else:
            return {
                'rec_pose': rec_pose,
                # 'rec_trans': rec_trans,
                'tar_pose': loaded_data["tar_pose"],
                'tar_exps': loaded_data["tar_exps"],
                'tar_beta': loaded_data["tar_beta"],
                'tar_trans': loaded_data["tar_trans"],
                # 'rec_exps': rec_exps,
            }
            
    def generate_weighted_tensor(self, a, b):
        bs, frame, dim = a.shape

        # 生成一个线性递减的权重向量，形状为 [frame]
        w = torch.linspace(1, 0, frame).to(a.device)  # 从1到0线性递减

        # 将权重向量扩展到 [bs, frame, dim] 的形状
        w = w.view(1, frame, 1).expand(bs, frame, dim)

        # 计算加权和
        c = a * w + b * (1 - w)

        return c

    def _g_test(self, loaded_data, all_in_one=True):
        mode = 'test'    
        bs, n, j = loaded_data["latent_face_top"].shape[0], loaded_data["latent_face_top"].shape[1], 1 
        tar_pose_face = loaded_data["tar_pose_face"]
        latent_face = loaded_data["latent_face_top"]
        tar_beta = loaded_data["tar_beta"]
        in_word = loaded_data["in_word"]
        tar_exps = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        in_audio = loaded_data["in_audio"]
        tar_trans = loaded_data["tar_trans"]
        joints = 1
        
        if False:
            net_out_val = self.model(
                in_audio = in_audio,
                in_word=in_word,
                mask=None, mode='test',
                in_motion = latent_face,
                in_id = loaded_data['tar_id'],
                use_attentions=True)
            latent_face_rec = net_out_val['latent_face_rec']
            #rec_pose_face = self.vq_model_face.decoder(latent_face_rec)
            _, rec_index_face, _, _ = self.vq_model_face.quantizer(latent_face_rec)
            rec_pose_face = self.vq_model_face.decode(rec_index_face)

            rec_pose = rc.rotation_6d_to_matrix(rec_pose_face[:, :, :joints*6].reshape(bs, n, joints, 6))
            tar_pose = rc.rotation_6d_to_matrix(tar_pose_face[:, :, :joints*6].reshape(bs, n, joints, 6))    
            rec_exps = rec_pose_face[:, :, joints*6:]            
            tar_exps = tar_pose_face[:, :, joints*6:]
            
            return {
                'rec_pose': rec_pose,
                'tar_pose': tar_pose,
                'tar_exps': tar_exps,
                'rec_exps': rec_exps,
                'tar_beta': loaded_data["tar_beta"],
                'tar_trans': loaded_data["tar_trans"],
            } 
                

        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames
        
        if remain != 0:
            tar_pose_face = tar_pose_face[:, :-remain, :]
            latent_face = latent_face[:, :-remain, :]      
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n = n - remain

        
        latent_face_list = []
        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            # audio fps is 16000 and pose fps is 30
            in_audio_tmp = in_audio[:, i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            # mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            # mask_val[:, :self.args.pre_frames, :] = 0.0
            latent_all_tmp = latent_face[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]

                # print(latent_all_tmp.shape, latent_last.shape)
                # latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            net_out_val = self.model(
                in_audio = in_audio_tmp,
                in_word=in_word_tmp,
                mask=None, mode='test',
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                use_attentions=True,)
            
            latent_face_rec = net_out_val['latent_face_rec']

            if i==0:
                latent_face_list.append(latent_face_rec[:, :-self.args.pre_frames, :])
            else:
                latent_face_rec[:, :self.args.pre_frames, :] = self.generate_weighted_tensor(latent_last[:, -self.args.pre_frames:, :], latent_face_rec[:, :self.args.pre_frames, :])
                latent_face_list.append(latent_face_rec[:, :-self.args.pre_frames, :])

            latent_last = latent_face_rec
        latent_face_list.append(latent_last[:, -self.args.pre_frames:, :])
        latent_face_all = torch.cat(latent_face_list, dim=1)

        _, rec_index_face, _, _ = self.vq_model_face.quantizer(latent_face_all)#no codebook 2. # this line.
        rec_pose_face = self.vq_model_face.decoder(rec_index_face)

        rec_pose = rc.rotation_6d_to_matrix(rec_pose_face[:, :, :joints * 6].reshape(bs, n, joints, 6))
        tar_pose = rc.rotation_6d_to_matrix(tar_pose_face[:, :, :joints * 6].reshape(bs, n, joints, 6))
        rec_exps = rec_pose_face[:, :, joints * 6:]
        tar_exps = tar_pose_face[:, :, joints * 6:]

        return {
            'rec_pose': rec_pose,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'rec_exps': rec_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
        }

    def train(self, epoch):
        #torch.autograd.set_detect_anomaly(True)
        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        # self.d_model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
    
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train', epoch)
            #with torch.autograd.detect_anomaly():
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # lr_d = self.opt_d.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
        # self.opt_d_s.step(epoch) 
    
    def val(self, epoch):
        self.model.eval()
        # self.d_model.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.train_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_training(loaded_data, False, 'val', epoch)
                
                if self.args.debug:
                    if its == 1: break
        # fid_motion = data_tools.FIDCalculator.frechet_distance(latent_out_motion_all, latent_ori_all)
        # self.tracker.update_meter("fid", "val", fid_motion)
        self.val_recording(epoch) 
    
    
    def test(self, epoch):
        
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], 1
                
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                    
                vertices_rec_face = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=rec_exps.reshape(bs*n, 100), 
                    jaw_pose=rec_pose, 
                    global_orient=torch.zeros(bs*n, 3).cuda(), 
                    body_pose=torch.zeros(bs*n, 21*3).cuda(), 
                    left_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
                    right_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
                    return_verts=True,
                    # return_joints=True,
                    leye_pose=torch.zeros(bs*n, 3).cuda(), 
                    reye_pose=torch.zeros(bs*n, 3).cuda(),
                )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose, 
                    global_orient=torch.zeros(bs*n, 3).cuda(), 
                    body_pose=torch.zeros(bs*n, 21*3).cuda(), 
                    left_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
                    right_hand_pose=torch.zeros(bs*n, 15*3).cuda(), 
                    return_verts=True,
                    # return_joints=True,
                    leye_pose=torch.zeros(bs*n, 3).cuda(), 
                    reye_pose=torch.zeros(bs*n, 3).cuda(),
                ) 
                
                
                #render 3 examples
                if (its % 10 == 0) and (not self.args.debug):
                    self.flame_dict['batch_size'] = bs*n
                    self.flame = FLAME(argparse.Namespace(**self.flame_dict)).to(self.rank).eval()
                    shape_params = torch.zeros(bs*n, 100).cuda()
                    rec_pose_params = torch.zeros(bs*n, 6).cuda()
                    rec_pose_params[:, 3:] = rec_pose
                    
                    tar_pose_params = torch.zeros(bs*n, 6).cuda()
                    tar_pose_params[:, 3:] = tar_pose
                    
                    rec_expression_params = rec_exps.reshape(bs*n, 100)
                    rec_flame_vertices, rec_flame_landmarks = self.flame(
                        shape_params, rec_expression_params, rec_pose_params
                    )
                    tar_expression_params = tar_exps.reshape(bs*n, 100)
                    tar_flame_vertices, tar_flame_landmarks = self.flame(
                        shape_params, tar_expression_params, tar_pose_params
                    )
                    
                    faces = self.flame.faces
                    
                    silent_video_file_path = utils.fast_render.generate_silent_videos_single(self.args.render_video_fps,
                                                                    self.args.render_video_width,
                                                                    self.args.render_video_height,
                                                                    1,#self.args.render_concurrent_num,
                                                                    self.args.render_tmp_img_filetype,
                                                                    bs*n, 
                                                                    rec_flame_vertices.detach().cpu().numpy(),
                                                                    tar_flame_vertices.detach().cpu().numpy(),
                                                                    faces,
                                                                    results_save_path)
                    audio_path = self.args.data_path + 'wave16k/' + test_seq_list.iloc[its]['id'] + '.wav'
                    base_filename_without_ext = test_seq_list.iloc[its]['id']
                    final_clip = os.path.join(results_save_path, f"{base_filename_without_ext}.mp4")
                    utils.media.add_audio_to_video(silent_video_file_path, audio_path, final_clip)
                    os.remove(silent_video_file_path)
                
              
                facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :n]
                facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :n]
                face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_rec[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
                l2 = self.reclatent_loss(facial_rec, facial_tar)
                l2_all += l2.item() * n
                lvel += face_vel_loss.item() * n
             
               
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"mygt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=np.zeros((bs*n, 3), dtype=float),
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=np.zeros((bs*n, 3), dtype=float),
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n
                if self.args.debug:
                  break

        logger.info(f"l2 loss: {l2_all/total_length}")
        logger.info(f"lvel loss: {lvel/total_length}")
        self.test_recording("ver_mse", l2_all/total_length, epoch)
        self.test_recording("ver_lvd", lvel/total_length, epoch)

        # data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")


    def test_demo(self, epoch):
        '''
        input audio and text, output motion
        do not calculate loss and metric
        save video
        '''
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path): 
            return 0
        os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        # self.eval_copy.eval()
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                # interpolate to 30fps  
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # print(rec_pose.shape, tar_pose.shape)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)

                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                        
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)

                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += n

        data_tools.result2target_vis(self.args.pose_version, results_save_path, results_save_path, self.test_demo, False)
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
