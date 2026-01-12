# Reference excerpt (not used by training):
# - Block diffusion mask creation and how it is passed into attention.
# - Source: models/dit.py in this repo.

from functools import partial

import torch
import torch.nn.functional as F

try:
  from torch.nn.attention.flex_attention import create_block_mask, flex_attention
  FLEX_ATTN_AVAILABLE = True
except Exception:
  FLEX_ATTN_AVAILABLE = False


def block_diff_mask(b, h, q_idx, kv_idx, block_size=None, n=None):
  # Indicate whether token belongs to xt or x0
  x0_flag_q = (q_idx >= n)
  x0_flag_kv = (kv_idx >= n)

  # Compute block indices
  block_q = torch.where(x0_flag_q == 1,
                        (q_idx - n) // block_size,
                        q_idx // block_size)
  block_kv = torch.where(x0_flag_kv == 1,
                         (kv_idx - n) // block_size,
                         kv_idx // block_size)

  # 1) Block Diagonal Mask (M_BD)
  block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)

  # 2) Offset Block-Causal Mask (M_OBC)
  offset_block_causal = (
    (block_q > block_kv)
    & (x0_flag_kv == 1)
    & (x0_flag_q == 0)
  )

  # 3) Block-Causal Mask (M_BC)
  block_causal = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)

  # 4) Combine Masks
  return block_diagonal | offset_block_causal | block_causal


def gen_mask(seqlen, block_size, attn_backend="sdpa"):
  # Builds a 2L x 2L mask for xt || x0.
  if attn_backend == "flex" and FLEX_ATTN_AVAILABLE:
    return create_block_mask(
      partial(block_diff_mask, block_size=block_size, n=seqlen),
      B=None, H=None, Q_LEN=seqlen * 2, KV_LEN=seqlen * 2)
  if attn_backend == "sdpa":
    return block_diff_mask(
      b=None, h=None,
      q_idx=torch.arange(seqlen * 2)[:, None],
      kv_idx=torch.arange(seqlen * 2)[None, :],
      block_size=block_size, n=seqlen)
  raise ValueError("Unknown attention backend")


def forward_mask_use(mask):
  # Reference use inside forward:
  # mask = self.block_diff_mask
  # x = block(..., mask=mask)
  return mask


def demo_attention_usage():
  # Simple SDPA demo: mask is 2L x 2L and broadcasts over batch/heads.
  torch.manual_seed(0)
  seq_len = 6
  block_size = 2
  total_len = seq_len * 2
  batch = 1
  heads = 2
  head_dim = 4
  q = torch.randn(batch, heads, total_len, head_dim)
  k = torch.randn(batch, heads, total_len, head_dim)
  v = torch.randn(batch, heads, total_len, head_dim)
  mask = gen_mask(seq_len, block_size, attn_backend="sdpa")
  out = F.scaled_dot_product_attention(
    q, k, v, attn_mask=mask, is_causal=False)
  return out, mask


def demo_flex_attention_usage():
  if not FLEX_ATTN_AVAILABLE:
    return None, None
  torch.manual_seed(0)
  seq_len = 6
  block_size = 2
  total_len = seq_len * 2
  batch = 1
  heads = 2
  head_dim = 4
  q = torch.randn(batch, heads, total_len, head_dim)
  k = torch.randn(batch, heads, total_len, head_dim)
  v = torch.randn(batch, heads, total_len, head_dim)
  mask = gen_mask(seq_len, block_size, attn_backend="flex")
  out = flex_attention(q, k, v, block_mask=mask)
  return out, mask


if __name__ == "__main__":
  out, mask = demo_attention_usage()
  print(f"mask.shape={tuple(mask.shape)} out.shape={tuple(out.shape)}")
  flex_out, flex_mask = demo_flex_attention_usage()
  if flex_out is None:
    print("flex_attention not available")
  else:
    print(f"flex_out.shape={tuple(flex_out.shape)} flex_mask={type(flex_mask)}")
