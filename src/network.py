import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import VOCAB_SIZE, MAX_SEQ_LENGTH, caption_to_tokens

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")


def sinusoidal_embedding(timesteps : torch.Tensor, embedding_dim : int):
    """
    Create sinusoidal embeddings for the given timesteps.

    Args:
        timesteps (torch.Tensor): A tensor of shape (batch_size,) containing the timesteps.
        embedding_dim (int): The dimension of the embedding.
    """
    half_dim = embedding_dim // 2
    temp = math.log(10000) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -temp)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)



class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), 
            nn.SiLU(), 
            nn.Linear(dim * 4, dim)
        )
 
    def forward(self, t):
        return self.mlp(sinusoidal_embedding(t, self.dim))
    
    

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len, n_layers=2, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, embedding_dim))
        layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dim_feedforward=embedding_dim * 4, batch_first=True, activation='gelu')
        # enable_nested_tensor=False : le fast-path "nested tensor" de PyTorch
        # n'est pas implémenté sur MPS et plante dès qu'un padding_mask est
        # fourni en mode eval (sample()/inférence).
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embedding_dim)
        
    def forward(self, tokens, padding_mask=None):
        x = self.embedding(tokens)
        x = x + self.positional_encoding[:, :x.size(1)]
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.norm(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, output_channels)  # pas un TimeMLP entier
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, input_channels),   # channels de x, pas de sortie
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels else nn.Identity()
        )

    def forward(self, x, t_emb):
        # t_emb : déjà calculé une fois dans TinyUNet.forward, passé en argument
        h = self.block1(x)                                    # norm+act+conv sur x brut
        h = h + self.time_proj(t_emb)[:, :, None, None]        # injection ICI, entre les 2 convs
        h = self.block2(h)
        return h + self.skip(x)
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, input_channels, output_channels, time_emb_dim, text_emb_dim, n_heads=4):
        super().__init__()
        self.time_proj = nn.Linear(time_emb_dim, output_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, input_channels),
            nn.SiLU(),
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_channels, kdim=text_emb_dim, vdim=text_emb_dim, num_heads=n_heads, batch_first=True
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        )
        self.skip = (
            nn.Conv2d(input_channels, output_channels, 1)
            if input_channels != output_channels else nn.Identity()
        )

    def forward(self, x, t_emb, text_emb, padding_mask=None):
        h = self.block1(x)
        h = h + self.time_proj(t_emb)[:, :, None, None]

        # Reshape for cross-attention
        B, C, H, W = h.shape
        h_flat = h.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)

        # Cross-attention: text_emb is (B, seq_len, text_emb_dim), one key/value per token
        attn_output, _ = self.cross_attention(h_flat, text_emb, text_emb, key_padding_mask=padding_mask)

        # Reshape back to (B, C, H, W)
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)
        
        h = h + attn_output
        h = self.block2(h)
        
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, img_channels=3, base_channels=48, time_emb_dim=128, context_dim=128):
        super().__init__()
        self.time_mlp = TimeMLP(time_emb_dim)
 
        self.in_conv = nn.Conv2d(img_channels, base_channels, kernel_size=3, padding=1)
 
        # descente
        self.down1 = ResBlock(base_channels, base_channels, time_emb_dim)               # 32x32
        self.pool1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)  # -> 16x16
 
        self.down2 = ResBlock(base_channels * 2, base_channels * 2, time_emb_dim)        # 16x16
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)  # -> 8x8
 
        # goulot avec cross-attention
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid_attn = CrossAttentionBlock(base_channels * 4, base_channels * 4, time_emb_dim, context_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
 
        # remontée -- attention aux channels après concat des skip connections
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1)  # -> 16x16
        self.up_res2 = ResBlock(base_channels * 4, base_channels * 2, time_emb_dim)  # in = up(base_ch*2) + skip(base_ch*2)
 
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)  # -> 32x32
        self.up_res1 = ResBlock(base_channels * 2, base_channels, time_emb_dim)  # in = up(base_ch) + skip(base_ch)
 
        self.out_norm = nn.GroupNorm(8, base_channels)
        self.out_conv = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
 
    def forward(self, x, t, context, padding_mask=None):
        t_emb = self.time_mlp(t)
 
        h0 = self.in_conv(x)
        h1 = self.down1(h0, t_emb)             # skip 1 : (B, base_ch, 32, 32)
        h1p = self.pool1(h1)
 
        h2 = self.down2(h1p, t_emb)            # skip 2 : (B, base_ch*2, 16, 16)
        h2p = self.pool2(h2)
 
        m = self.mid1(h2p, t_emb)
        m = self.mid_attn(m, t_emb, context, padding_mask=padding_mask)
        m = self.mid2(m, t_emb)
 
        u2 = self.up2(m)
        u2 = self.up_res2(torch.cat([u2, h2], dim=1), t_emb)
 
        u1 = self.up1(u2)
        u1 = self.up_res1(torch.cat([u1, h1], dim=1), t_emb)
 
        return self.out_conv(F.silu(self.out_norm(u1)))

class TextToImageModel(nn.Module):
    """Le "gros" modèle : encodeur texte + U-Net assemblés. flow_matching.py /
    ddpm.py n'appelleront que forward(x_t, t, tokens, padding_mask) sans se
    soucier de ce qu'il y a dedans."""
 
    def __init__(self, vocab_size, max_seq_len, embedding_dim=128,
                 time_emb_dim=128, base_channels=48, img_channels=3):
        super().__init__()
        self.text_encoder = TextEncoder(vocab_size, embedding_dim, max_seq_len)
        self.unet = TinyUNet(
            img_channels=img_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
            context_dim=embedding_dim,
        )
 
    def forward(self, x_t, t, tokens, padding_mask=None):
        context = self.text_encoder(tokens, padding_mask=padding_mask)
        return self.unet(x_t, t, context, padding_mask=padding_mask)
    
    
def check_shape(name, actual, expected):
    assert actual == expected, f"{name}: expected {expected}, got {actual}"
    print(f"{name} OK: {actual}")


if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        config = json.load(f)

    general_cfg = config["general"]
    time_mlp_cfg = config["time_mlp"]
    text_encoder_cfg = config["text_encoder"]
    unet_cfg = config["unet"]
    text_to_image_model_cfg = config["text_to_image_model"]

    batch_size = general_cfg["batch_size"]
    img_size = general_cfg["img_size"]
    img_channels = general_cfg["img_channels"]
    embedding_dim = general_cfg["embedding_dim"]
    time_emb_dim = general_cfg["time_emb_dim"]
    vocab_size = VOCAB_SIZE
    max_seq_len = MAX_SEQ_LENGTH

    # sinusoidal_embedding
    timesteps = torch.randint(0, 1000, (batch_size,))
    emb = sinusoidal_embedding(timesteps, embedding_dim)
    check_shape("sinusoidal_embedding", tuple(emb.shape), (batch_size, embedding_dim))

    # TimeMLP
    time_mlp = TimeMLP(**time_mlp_cfg)
    t_emb = time_mlp(timesteps)
    check_shape("TimeMLP", tuple(t_emb.shape), (batch_size, time_mlp_cfg["dim"]))

    # TextEncoder -- built from a real captioned sample out of data.py
    text_encoder = TextEncoder(vocab_size=vocab_size, max_seq_len=max_seq_len, **text_encoder_cfg)
    single_tokens, single_padding_mask = caption_to_tokens(color_name="red", digit=7, angle=42)
    tokens = single_tokens.unsqueeze(0).repeat(batch_size, 1)
    padding_mask = single_padding_mask.unsqueeze(0).repeat(batch_size, 1)
    context = text_encoder(tokens, padding_mask)
    check_shape("TextEncoder", tuple(context.shape), (batch_size, max_seq_len, text_encoder_cfg["embedding_dim"]))

    # ResBlock -- with and without a channel change
    base_channels = unet_cfg["base_channels"]
    x32 = torch.randn(batch_size, base_channels, img_size, img_size)
    res_same = ResBlock(base_channels, base_channels, time_emb_dim)
    res_same_out = res_same(x32, t_emb)
    check_shape("ResBlock (same channels)", tuple(res_same_out.shape), (batch_size, base_channels, img_size, img_size))

    res_up = ResBlock(base_channels, base_channels * 2, time_emb_dim)
    res_up_out = res_up(x32, t_emb)
    check_shape("ResBlock (channel change)", tuple(res_up_out.shape), (batch_size, base_channels * 2, img_size, img_size))

    # CrossAttentionBlock
    cross_attn = CrossAttentionBlock(base_channels, base_channels * 2, time_emb_dim, unet_cfg["context_dim"])
    cross_attn_out = cross_attn(x32, t_emb, context, padding_mask=padding_mask)
    check_shape("CrossAttentionBlock", tuple(cross_attn_out.shape), (batch_size, base_channels * 2, img_size, img_size))

    # TinyUNet
    unet = TinyUNet(img_channels=img_channels, **unet_cfg)
    x_t = torch.randn(batch_size, img_channels, img_size, img_size)
    unet_out = unet(x_t, timesteps, context, padding_mask=padding_mask)
    check_shape("TinyUNet", tuple(unet_out.shape), (batch_size, img_channels, img_size, img_size))

    # TextToImageModel -- full pipeline, tokens straight in, image straight out
    model = TextToImageModel(vocab_size=vocab_size, max_seq_len=max_seq_len, **text_to_image_model_cfg)
    model_out = model(x_t, timesteps, tokens, padding_mask=padding_mask)
    check_shape("TextToImageModel", tuple(model_out.shape), (batch_size, img_channels, img_size, img_size))

    print("All shape checks passed.")