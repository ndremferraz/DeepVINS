import torch 
import torch.nn as nn

from encoder import ImageEncoder, InertialEncoder

class AttentionBlock(nn.Module):
  def __init__(self, embed_dim=1024, num_heads=4, dropout=0.1):
      super().__init__()

      self.norm1 = nn.LayerNorm(embed_dim)
      self.dropout = nn.Dropout(dropout)

      self.attn = nn.MultiheadAttention(
          embed_dim=embed_dim,
          num_heads=num_heads,
          dropout=dropout,
          batch_first=True
      )

      self.norm2 = nn.LayerNorm(embed_dim)
      self.l2 = nn.Sequential(
          nn.Linear(embed_dim, 4 * embed_dim),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(4 * embed_dim, embed_dim),
          nn.Dropout(dropout)
      )

  def forward(self, x):
      t = x.size(1)
      causal_mask = torch.triu(
            torch.ones(t, t, device=x.device, dtype=torch.bool),
            diagonal=1
      )

      attn_input = self.norm1(x)
      attn_output, _ = self.attn(
          query=attn_input,
          key=attn_input,
          value=attn_input,
          attn_mask=causal_mask,
          need_weights=False
      )
      x = x + self.dropout(attn_output)
      mlp_input = self.norm2(x)
      mlp_output = self.l2(mlp_input)
      x = x + self.dropout(mlp_output)

      return x
  
class CausalFusionModel(nn.Module):

  def __init__(
      self,
      attn_blocks: int = 4,
      embed_dim: int = 1024,
      num_heads: int = 4,
      context_length: int = 64,
      output_dim: int = 7,
      dropout: float = 0.1
  ):

    super().__init__()

    self.imu_encoder = InertialEncoder()
    self.img_encoder = ImageEncoder()

    self.pos_emb = nn.Embedding(context_length, embed_dim)

    self.attn_blocks = nn.ModuleList(
        [AttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        for _ in range(attn_blocks)]
    )

    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, output_dim)
    )
    self._init_weights()

  def _init_weights(self):
      for module in self.modules():
          if isinstance(module, (nn.Conv1d, nn.Conv2d)):
              nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
              if module.bias is not None:
                  nn.init.zeros_(module.bias)
          elif isinstance(module, nn.Linear):
              nn.init.xavier_uniform_(module.weight)
              if module.bias is not None:
                  nn.init.zeros_(module.bias)
          elif isinstance(module, nn.MultiheadAttention):
              nn.init.xavier_uniform_(module.in_proj_weight)
              if module.in_proj_bias is not None:
                  nn.init.zeros_(module.in_proj_bias)
              nn.init.xavier_uniform_(module.out_proj.weight)
              if module.out_proj.bias is not None:
                  nn.init.zeros_(module.out_proj.bias)
          elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
              if module.weight is not None:
                  nn.init.ones_(module.weight)
              if module.bias is not None:
                  nn.init.zeros_(module.bias)


  def forward(self, img_batch: torch.Tensor, imu_batch: torch.Tensor):
    img_out = self.img_encoder(img_batch)
    imu_out = self.imu_encoder(imu_batch)

    tokens = torch.cat((imu_out, img_out), dim=-1)
    seq_len = tokens.shape[1]

    pos = torch.arange(seq_len, device=tokens.device)
    pos_emb = self.pos_emb(pos)

    tokens = tokens + pos_emb

    for attn_block in self.attn_blocks:
        tokens = attn_block(tokens)

    tokens = self.norm(tokens)
    tokens = self.head(tokens)

    return tokens