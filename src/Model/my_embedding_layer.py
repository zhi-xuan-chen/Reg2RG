import torch.nn as nn
import torch.nn.functional as F
import torch
from .helpers import PerceiverResampler
from .utils import get_visual_encoder
from einops import rearrange, repeat
from einops_exts import rearrange_many
import torchvision
from .vit_3d import ViT
from einops.layers.torch import Rearrange
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from torch.utils.checkpoint import checkpoint
from torch.autograd import Variable
import random
from transformers import AutoTokenizer, AutoModel
from monai.networks.nets.swin_unetr import SwinTransformer
from .cross_attention import TwoWayTransformer

CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]

REGIONS = [
    'abdomen',
    'bone',
    'breast',
    'esophagus',
    'heart',
    'lung',
    'mediastinum',
    'pleura',
    'thyroid',
    'trachea and bronchie',
]

class MyEmbedding(nn.Module):
    def __init__(self, pretrained_visual_encoder=None, pretrained_adapter=None, num_embeddings=32000, embedding_dim=4096, perceiver_num=32, vis_dim=768, patch_size=32, frame_patch_size=4, seg_channel=256):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.torch.randn((num_embeddings, embedding_dim)), requires_grad=True)  # NOTE: will be initialized using the weight from MedLLaMA
        self.image_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.region_token_weight = nn.Parameter(
            torch.randn((2, embedding_dim)), requires_grad=True)
        self.patch_size = patch_size
        self.frame_patch_size = frame_patch_size
        self.seg_channel = seg_channel

        self.vision_encoder = ViT(
            image_size=512,          # image size
            frames=512,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=frame_patch_size,      # frame patch size
            dim=vis_dim,
            depth=12,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

        self.mask_encoder = ViT(
            image_size=256,          # image size
            frames=64,               # max number of frames
            image_patch_size=patch_size,     # image patch size
            frame_patch_size=16,      # frame patch size
            dim=255,
            depth=3,
            heads=8,
            mlp_dim=512,
            channels = 1,
            dropout=0.1,
            emb_dropout=0.1
        )

        # load pretrained vision encoder from RadFM
        if pretrained_visual_encoder is not None:
            vit3d_ckpt = torch.load(pretrained_visual_encoder, map_location='cpu')
            self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        # vit3d_ckpt = torch.load(
        #     '/data/chenzhixuan/checkpoints/LLM4CTRG/t3d_swin3d.pth', map_location='cpu')
        # self.vision_encoder.load_state_dict(vit3d_ckpt, strict=True)
        # print('load pretrained vision encoder from T3D')

        # frozen the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        self.vis_dim = vis_dim

        self.perceiver = PerceiverResampler(
            dim=self.vis_dim, num_latents=perceiver_num)
        # load pretrained perceiver and fc from RadFM
        if pretrained_adapter is not None:
            state_dict = torch.load(pretrained_adapter, map_location='cpu')
            self.perceiver.load_state_dict(state_dict['perceiver'])
            # self.fc.load_state_dict(state_dict['fc'])
        
        # self.cross_attn = TwoWayTransformer(
        #     depth=3,
        #     embedding_dim=self.vis_dim,
        #     num_heads=8,
        #     mlp_dim=1024,
        # )
        
        self.fc = nn.Linear(self.vis_dim, self.embedding_dim)
        self.mask_fc = nn.Linear(255, self.embedding_dim)

        
    def forward(self, vision_x, mask_x, text_input, region2areas):
        # 获取输入张量vision_x的形状，并将各个维度的大小赋值给相应的变量
        # B: 批量大小，一次处理的样本数量
        # S: 序列长度，例如在处理文本时，这可能是句子的长度
        # C: 通道数，例如在处理图像时，这可能是颜色通道的数量（红、绿、蓝）
        # H: 高度，例如在处理图像时，这可能是图像的高度
        # W: 宽度，例如在处理图像时，这可能是图像的宽度
        # D: 深"度，例如在处理三维数据（如3D图像或点云）时，这可能是数据的深度

        B, S, C, H, W, D = next(iter(vision_x.values())).shape
        
        vision_temp = vision_x['image']
        vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
        vision_temp, pos_embedding = self.vision_encoder(vision_temp)
        vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
        vision_temp = vision_temp.unsqueeze(2)
        vision_temp = self.perceiver(vision_temp)
        n = vision_temp.shape[2]
        vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
        vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n*S)
        image_embedding = vision_temp
        
        del vision_x['image']
    
        region_embeddings = vision_x
        mask_embeddings = mask_x
                
        for key in region_embeddings.keys():
            vision_temp = region_embeddings[key]
            vision_temp = rearrange(vision_temp, "b S c h w d-> (b S) c h w d")
            vision_temp, _ = self.vision_encoder(vision_temp)
            vision_temp = rearrange(vision_temp, "(b s) v d -> b s v d", b=B, s=S)
            vision_temp = vision_temp.unsqueeze(2)
            vision_temp = self.perceiver(vision_temp)
            n = vision_temp.shape[2]
            vision_temp = rearrange(vision_temp, "b s n d -> (b s n) d")
            vision_temp = rearrange(vision_temp, "(b T) d -> b T d", b=B, T=n*S)
            
            region_embeddings[key] = vision_temp

            mask_embedding, _ = self.mask_encoder(mask_x[key])
            mask_embedding = torch.mean(mask_embedding, dim=1)
            mask_embeddings[key] = mask_embedding

        # combined_region_embedding = torch.cat([region_embeddings[key].unsqueeze(1) for key in region_embeddings.keys()], dim=1)
        
        # image_embedding, _ = self.cross_attn(
        #     image_embedding=image_embedding, 
        #     region_embedding=combined_region_embedding, 
        #     )
        
        image_embedding = self.fc(image_embedding)
        
        for key in region_embeddings.keys():
            region_embeddings[key] = self.fc(region_embeddings[key])
            mask_embeddings[key] = self.mask_fc(mask_embeddings[key])

        # fuse region embeddings and mask embeddings
        for key in region_embeddings.keys():
            region_embeddings[key] = torch.cat([region_embeddings[key], mask_embeddings[key].unsqueeze(1)], dim=1) 
        
        max_region = len(region_embeddings)  
        # zero init the vision embedding
        vision_region_embedding = torch.zeros(
            (B, 33*max_region, self.embedding_dim), device=text_input.device) # NOTE: 2 means 1 region and 1 mask embeddings
        
        for i in range(B):
            for j in range(len(region2areas[i])):
                region = region2areas[i][j]
                vision_region_embedding[i, j*33:(j+1)*33, :] = region_embeddings[region][i, :, :]

        embedding_weight = torch.cat([self.weight, self.image_token_weight, self.region_token_weight], dim=0)  # num_embeddings+2, embedding_dim
        embedding_weight = embedding_weight.unsqueeze(0).repeat(B, 1, 1)  # B, num_embeddings+2, embedding_dim
        # B, num_embeddings+2+n, embedding_dim

        embedding_weight = torch.cat([embedding_weight, image_embedding, vision_region_embedding], dim=1)
        text_input = F.one_hot(text_input, embedding_weight.shape[1]).to(
            vision_region_embedding.dtype).to(text_input.device)  # B, N, num_embeddings+2+n
        out_put = torch.matmul(text_input, embedding_weight)
        
        return out_put
