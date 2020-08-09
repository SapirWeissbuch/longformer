import torch
import torch.nn.functional as F
from longformer.diagonaled_mm_tvm import mask_invalid_locations

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True


def _unfold_conv(base, size, step):
    d0, d1, d2 = base.shape
    reshape_base = base.reshape([d0,d1//step, step,d2])
    transpose_base = reshape_base.permute(3, 2, 0 ,1)
    filter = torch.eye(size, device=xm.xla_device()).view([1,size//step,step,size]).permute(3,2,0,1)
    res = torch.nn.functional.conv2d(transpose_base, filter)
    return res.permute(2,3,1,0)


def _skew(x, direction, padding_value):
    '''Convert diagonals into columns (or columns into diagonals depending on `direction`'''
    x_padded = F.pad(x, direction, value=padding_value)
    x_padded = x_padded.view(*x_padded.size()[:-2], x_padded.size(-1), x_padded.size(-2))
    return x_padded


def _skew2(x, padding_value):
    '''shift every row 1 step to right converting columns into diagonals'''
    # X = B x C x M x L
    B, C, M, L = x.size()
    x = F.pad(x, (0, M + 1), value=padding_value)  # B x C x M x (L+M+1)
    x = x.view(B, C, -1)  # B x C x ML+MM+M
    x = x[:, :, :-M]  # B x C x ML+MM
    x = x.view(B, C, M, M + L)  # B x C, M x L+M
    x = x[:, :, :, :-1]
    return x


def _chunk(x, w):
    '''convert into overlapping chunkings. Chunk size = 2w, overlap size = w'''

    # non-overlapping chunks of size = 2w
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))

    # use `as_strided` to make the chunks overlap with an overlap size = w
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1

    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def sliding_chunks_matmul_qk_3(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % w == 0
    assert q.size() == k.size()
    # chunk seqlen into non-overlapping chunks of size w
    chunk_q = q.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k = k.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_q_expanded = torch.stack((chunk_q.roll(shifts=-1, dims=1), chunk_q, chunk_q.roll(shifts=1, dims=1)), dim=-1)
    chunk_q_expanded = torch.stack((chunk_q, chunk_q, chunk_q), dim=-1)
    diagonal_attn = torch.einsum('bcxhde,bcyhd->bcxhey', (chunk_q_expanded, chunk_k))  # multiply
    diagonal_attn[:, :, :, :, 0] = F.pad(diagonal_attn[:, :-1, :, :, 0], (0, 0, 0, 0, 0, 0, 1, 0), value=-float('inf'))
    diagonal_attn[:, :, :, :, 2] = F.pad(diagonal_attn[:, 1:, :, :, 2], (0, 0, 0, 0, 0, 0, 0, 1), value=-float('inf'))
    return diagonal_attn.reshape(bsz, seqlen, num_heads, 3 * w)


def sliding_chunks_matmul_qk_2(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % w == 0
    assert q.size() == k.size()
    # chunk seqlen into non-overlapping chunks of size w
    chunk_q = q.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_k = k.view(bsz, seqlen // w, w, num_heads, head_dim)

    chunk_attn_main_diagonal = torch.einsum('bcxhd,bcyhd->bcxhy', (chunk_q, chunk_k))  # multiply

    chunk_attn_upper_diagonal = torch.einsum('bcxhd,bcyhd->bcxhy', (chunk_q.roll(shifts=1, dims=1), chunk_k))
    chunk_attn_upper_diagonal[:, 0] = -float('inf')
    chunk_attn_upper_diagonal = chunk_attn_upper_diagonal.roll(shifts=-1, dims=1)
    upper_diagonal_mask = q.new_ones((w, w)).tril()[None, None, :, None].bool()
    chunk_attn_upper_diagonal.masked_fill_(~upper_diagonal_mask, -float('inf'))

    chunk_attn_lower_diagonal = torch.einsum('bcxhd,bcyhd->bcxhy', (chunk_q.roll(shifts=-1, dims=1), chunk_k))
    chunk_attn_lower_diagonal[:, -1] = -float('inf')
    chunk_attn_lower_diagonal = chunk_attn_lower_diagonal.roll(shifts=1, dims=1)
    lower_diagonal_mask = q.new_ones((w, w)).triu()[None, None, :, None].bool()
    chunk_attn_lower_diagonal.masked_fill_(~lower_diagonal_mask, -float('inf'))

    diagonal_attn = torch.cat((chunk_attn_lower_diagonal, chunk_attn_main_diagonal, chunk_attn_upper_diagonal), -1)
    return diagonal_attn.view(bsz, seqlen, num_heads, 3 * w)


def sliding_chunks_matmul_qk(q: torch.Tensor, k: torch.Tensor, w: int, padding_value: float):
    '''Matrix multiplicatio of query x key tensors using with a sliding window attention pattern.
    This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
    with an overlap of size w'''
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    assert q.size() == k.size()

    chunks_count = seqlen // w - 1

    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size w * 2
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    if XLA_AVAILABLE:
        chunk_q = _unfold_conv(q, 2 * w, w)
        chunk_k = _unfold_conv(k, 2 * w, w)
    else:
        chunk_q = _chunk(q, w)
        chunk_k = _chunk(k, w)

    # matrix multipication
    # bcxd: bsz*num_heads x chunks x 2w x head_dim
    # bcyd: bsz*num_heads x chunks x 2w x head_dim
    # bcxy: bsz*num_heads x chunks x 2w x 2w
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))  # multiply

    # convert diagonals into columns
    diagonal_chunk_attn = _skew(chunk_attn, direction=(0, 0, 0, 1), padding_value=padding_value)

    # allocate space for the overall attention matrix where the chunks are compined. The last dimension
    # has (w * 2 + 1) columns. The first (w) columns are the w lower triangles (attention from a word to
    # w previous words). The following column is attention score from each word to itself, then
    # followed by w columns for the upper triangle.

    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))

    # copy parts from diagonal_chunk_attn into the compined matrix of attentions
    # - copying the main diagonal and the upper triangle
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    # - copying the lower triangle
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, - (w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]

    # separate bsz and num_heads dimensions again
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)

    mask_invalid_locations(diagonal_attn, w, 1, False)
    return diagonal_attn


def sliding_chunks_matmul_pv_3(prob: torch.Tensor, v: torch.Tensor, w: int):
    bsz, seqlen, num_heads, head_dim = v.size()
    chunk_prob = prob.view(bsz, seqlen // w, w, num_heads, 3, w)
    chunk_v = v.view(bsz, seqlen // w, w, num_heads, head_dim)
    chunk_v_extended = torch.stack((chunk_v.roll(shifts=1, dims=1), chunk_v, chunk_v.roll(shifts=-1, dims=1)), dim=-1)
    chunk_v_extended = torch.stack((chunk_v, chunk_v, chunk_v), dim=-1)
    context = torch.einsum('bcwhpd,bcdhep->bcwhe', (chunk_prob, chunk_v_extended))
    return context.reshape(bsz, seqlen, num_heads, head_dim)


def sliding_chunks_matmul_pv_2(prob: torch.Tensor, v: torch.Tensor, w: int):
    bsz, seqlen, num_heads, head_dim = v.size()
    chunk_prob = prob.view(bsz, seqlen // w, w, num_heads, 3, w)
    chunk_v = v.view(bsz, seqlen // w, w, num_heads, head_dim)
    context = torch.einsum('bcwhd,bcdhe->bcwhe', (chunk_prob[:, :, :, :, 1], chunk_v))
    context += torch.einsum('bcwhd,bcdhe->bcwhe', (chunk_prob[:, :, :, :, 0], chunk_v.roll(shifts=1, dims=1)))
    context += torch.einsum('bcwhd,bcdhe->bcwhe', (chunk_prob[:, :, :, :, 2], chunk_v.roll(shifts=-1, dims=1)))
    return context.reshape(bsz, seqlen, num_heads, head_dim)


def sliding_chunks_matmul_pv(prob: torch.Tensor, v: torch.Tensor, w: int):
    '''Same as sliding_chunks_matmul_qk but for prob and value tensors. It is expecting the same output
    format from sliding_chunks_matmul_qk'''
    bsz, seqlen, num_heads, head_dim = v.size()
    assert seqlen % (w * 2) == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * w + 1
    chunks_count = seqlen // w - 1
    # group bsz and num_heads dimensions into one, then chunk seqlen into chunks of size 2w
    chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)

    # group bsz and num_heads dimensions into one
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)

    # pad seqlen with w at the beginning of the sequence and another w at the end
    padded_v = F.pad(v, (0, 0, w, w), value=-1)

    # chunk padded_v into chunks of size 3w and an overlap of size w
    if XLA_AVAILABLE:
        chunk_v = _unfold_conv(padded_v, 3 * w, w)
    else:
        chunk_v_size = (bsz * num_heads, chunks_count + 1, 3 * w, head_dim)
        chunk_v_stride = padded_v.stride()
        chunk_v_stride = chunk_v_stride[0], w * chunk_v_stride[1], chunk_v_stride[1], chunk_v_stride[2]
        chunk_v = padded_v.as_strided(size=chunk_v_size, stride=chunk_v_stride)

    skewed_prob = _skew2(chunk_prob, padding_value=0)

    context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


def pad_to_window_size(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       one_sided_window_size: int, pad_token_id: int):
    '''A helper function to pad tokens and mask to work with the sliding_chunks implementation of Longformer selfattention.
    Input:
        input_ids = torch.Tensor(bsz x seqlen): ids of wordpieces
        attention_mask = torch.Tensor(bsz x seqlen): attention mask
        one_sided_window_size = int: window size on one side of each token
        pad_token_id = int: tokenizer.pad_token_id
    Returns
        (input_ids, attention_mask) padded to length divisible by 2 * one_sided_window_size
    '''
    w = 2 * one_sided_window_size
    seqlen = input_ids.size(1)
    padding_len = (w - seqlen % w) % w
    input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
    attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
    return input_ids, attention_mask
