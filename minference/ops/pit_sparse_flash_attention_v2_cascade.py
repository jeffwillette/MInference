# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import torch
import triton
import triton.language as tl

from minference.cuda import convert_vertical_slash_indexes


# @triton.autotune(
#    configs=[
#        triton.Config({}, num_stages=1, num_warps=4),
#        triton.Config({}, num_stages=1, num_warps=8),
#        triton.Config({}, num_stages=2, num_warps=4),
#        triton.Config({}, num_stages=2, num_warps=8),
#        triton.Config({}, num_stages=3, num_warps=4),
#        triton.Config({}, num_stages=3, num_warps=8),
#        triton.Config({}, num_stages=4, num_warps=4),
#        triton.Config({}, num_stages=4, num_warps=8),
#        triton.Config({}, num_stages=5, num_warps=4),
#        triton.Config({}, num_stages=5, num_warps=8),
#    ],
#    key=['N_CTX'],
# )


@triton.jit
def _triton_mixed_sparse_attn_fwd_kernel(
    CK,
    stride_ckz, stride_ckh, stride_ckn, stride_ckk,
    CV,
    stride_cvz, stride_cvh, stride_cvn, stride_cvk,
    N_CKV,
    MASK,
    stride_mm, stride_mn,
    SCORES,
    stride_sz, stride_sh, stride_sn,  # original args
    Q, K, V, seqlens, sm_scale,
    block_count, block_offset, column_count, column_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX: tl.constexpr,
    NUM_ROWS, NNZ_S, NNZ_V,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen

    # copied in from the cascade flash attention kernel. =================
    tl.static_assert(BLOCK_N <= BLOCK_DMODEL)
    off_z = off_hz // H
    off_h = off_hz % H

    ck_offset = off_z.to(tl.int64) * stride_ckz + off_h.to(tl.int64) * stride_ckh
    cv_offset = off_z.to(tl.int64) * stride_cvz + off_h.to(tl.int64) * stride_cvh
    score_offset = off_z.to(tl.int64) * stride_sz + off_h.to(
        tl.int64) * stride_sh

    # block pointers
    ck_offset = ck_offset + (offs_d[:, None]).to(tl.int64) * stride_ckk
    cv_offset = cv_offset + (offs_d[None, :]).to(tl.int64) * stride_cvk

    Mask_block = offs_n * stride_mn
    beta = 0.999

    # loop over k, v and update accumulator
    # mask_vert = tl.full((BLOCK_M, 1), value=1, dtype=tl.int1)
    # print("m mask: ", m_mask)
    for start_n in range(0, N_CKV, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        cols = start_n + offs_n
        n_mask = cols < N_CKV
        k = tl.load(CK + ck_offset + cols[None, :] * stride_ckn, mask=n_mask[None, :], other=0.0)
        v = tl.load(CV + cv_offset + cols[:, None] * stride_cvn, mask=n_mask[:, None], other=0.0)

        # -- compute qk ----
        # k = tl.load(CK_block_ptr)

        # load from the cache mask (sink, keyvals)
        mask = tl.load(MASK + Mask_block)[None, :]
        # print("mask: ", mask)
        mask = tl.where(mask, 0, 1)

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # qk = tl.where(mask, float("-inf"), qk)
        # qk = tl.where(m_mask & mask, qk, float("-inf"))
        qk = tl.dot(q, k)
        # qk += tl.where(mask, -1.0e6, 0)
        qk += tl.where(m_mask & mask, 0.0, -1.0e6)

        # ------------------------------
        # for sum accumulation
        # coeff = 1

        # for EMA accumulation
        exps = tl.flip(tl.arange(0, BLOCK_M))[:, None]  # original submission
        # exps = N_CTX - (start_m * tl.flip(tl.arange(0, BLOCK_M))[:, None])  # bugfix?
        unmasked = tl.where(mask, 0, 1)
        exps = tl.exp2(exps.to(dtype) * tl.log2(beta))
        coeff = exps * (1 - beta) * unmasked
        # ------------------------------

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        # this will be incrementally more accurate as we get to the end of the sequence.
        # it is not exactly equivalent to the non-flash attention version.
        # print("qk term: ", tl.sum((p / l_ij[:, None])))
        # print("coeff term: ", coeff)
        steps_left = (N_CKV - (start_n + BLOCK_N)) // BLOCK_N
        steps_done = (start_n + BLOCK_N) // BLOCK_N
        adj = steps_left / steps_done
        # print("adj: ", adj)
        score_offset_inner = (start_n + tl.arange(0, BLOCK_N)).to(tl.int64) * stride_sn
        tl.atomic_add(SCORES + score_offset + score_offset_inner, val=tl.sum((p / (l_i[:, None] + (l_i[:, None] * adj) + 1e-6)) * coeff, 0))

        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        # v = tl.load(CV_block_ptr)

        acc += tl.dot(p.to(dtype), v)
        # update m_i and l_i
        m_i = m_ij

        Mask_block += BLOCK_N * stride_mn

    # dense kernel test
    # for start_n in range(0, N_CTX, BLOCK_N):
    #     cols = start_n + offs_n
    #     n_mask = cols < seqlen
    #     k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
    #     v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)

    #     qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    #     causal_mask = cols[None, :] <= offs_m[:, None]
    #     qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
    #     qk += tl.dot(q, k)

    #     # -- compute scaling constant --
    #     m_i_new = tl.maximum(m_i, tl.max(qk, 1))
    #     alpha = tl.math.exp2(m_i - m_i_new)
    #     p = tl.math.exp2(qk - m_i_new[:, None])
    #     # -- scale and update acc --
    #     acc_scale = l_i * 0 + alpha  # workaround some compiler bug
    #     acc *= acc_scale[:, None]
    #     acc += tl.dot(p.to(dtype), v)
    #     # -- update m_i and l_i --
    #     l_i = l_i * alpha + tl.sum(p, 1)
    #     m_i = m_i_new

    for block_index in range(num_blks):
        start_n = tl.load(blks_ptr + block_index)
        cols = start_n + offs_n
        n_mask = cols < seqlen
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    for start_n2 in range(0, num_cols, BLOCK_N):
        n_mask = start_n2 + offs_n < num_cols
        cols = tl.load(cols_ptr + start_n2 + offs_n, mask=n_mask, other=0)
        # -- load k, v --
        k = tl.load(k_ptrs + cols[None, :] * stride_kn, mask=n_mask[None, :], other=0.0)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn, mask=n_mask[:, None], other=0.0)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.where(m_mask & n_mask, qk, float("-inf"))
        qk += tl.dot(q, k)
        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]
        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    # acc = tl.where(m_mask, acc / l_i[:, None], 0.0)
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_mixed_sparse_attention(
    ck: torch.Tensor,
    cv: torch.Tensor,
    cmask: torch.Tensor,
    q: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v: torch.Tensor,          # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens: torch.Tensor,    # [BATCH, ]
    block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    sm_scale: float,
    block_size_M: int = 64,
    block_size_N: int = 64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    grid = (triton.cdiv(q.shape[2], block_size_M), q.shape[0] * q.shape[1], 1)
    dtype = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

    N_CKV = ck.size(2)
    scores = torch.zeros(q.size(0),
                         q.size(1),
                         ck.size(-2),
                         device=q.device,
                         dtype=q.dtype)

    # print(f"{cmask.stride()=}")
    # print(f"before kernel {N_CKV=}")
    # print(f"before kernel {ck.stride()=} {cv.stride()=}")

    _triton_mixed_sparse_attn_fwd_kernel[grid](
        ck,
        ck.stride(0), ck.stride(1), ck.stride(2), ck.stride(3),
        cv,
        cv.stride(0), cv.stride(1), cv.stride(2), cv.stride(3),
        N_CKV,
        cmask,
        cmask.stride(0), cmask.stride(1),
        scores,
        scores.stride(0), scores.stride(1), scores.stride(2),
        q, k, v, seqlens, sm_scale,
        block_count, block_offset, column_count, column_index,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        block_count.shape[-1], block_offset.shape[-1], column_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o


def vertical_slash_sparse_attention_cascade(
    ck: torch.Tensor,
    cv: torch.Tensor,
    cmask: torch.Tensor,
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    s_idx: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    block_size_M: int = 64,
    block_size_N: int = 64,
):
    batch_size, num_heads, context_size, head_dim = query.shape
    pad = block_size_M - (context_size & (block_size_M - 1))
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])

    if head_dim not in [16, 32, 64, 128, 256, 512]:
        target_dim = 2 ** math.ceil(math.log2(head_dim)) - head_dim
        query = torch.nn.functional.pad(query, [0, target_dim, 0, 0, 0, 0, 0, 0])
        key = torch.nn.functional.pad(key, [0, target_dim, 0, 0, 0, 0, 0, 0])
        value = torch.nn.functional.pad(value, [0, target_dim, 0, 0, 0, 0, 0, 0])

    v_idx = v_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=False)[0]
    s_idx = s_idx.to(torch.int32).reshape((batch_size, num_heads, -1)).sort(dim=-1, descending=True)[0]
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=query.device)
    sm_scale = head_dim ** -0.5
    block_count, block_offset, column_count, column_index = convert_vertical_slash_indexes(
        seqlens, v_idx, s_idx, context_size, block_size_M, block_size_N,
    )

    cmask = cmask[None, :].contiguous()
    assert len(cmask.shape) == 2
    out = _triton_mixed_sparse_attention(
        ck, cv, cmask,
        query, key, value, seqlens,
        block_count, block_offset, column_count, column_index,
        sm_scale, block_size_M, block_size_N,
    )
    return out[..., :context_size, :head_dim]


if __name__ == "__main__":
    import math

    last_q = 64
    arange = torch.arange(last_q, device="cuda")
    LAST_Q_MASK = arange[None, None, :, None] >= arange[None, None, None, :]
    ROPE_TYPE = None
    SEARCH_MASK = None

    print("testing vertical slash sparse cascade")
    NQ, NKV, D = 8192, 8192, 64
    N_VIDX, N_SIDX = 200, 200
    ck = torch.randn(1, 32, NKV, D).to(torch.float16).cuda()
    cv = torch.randn(1, 32, NKV, D).to(torch.float16).cuda()
    cmask = torch.rand(10, NKV).cuda() > 0.5

    q = torch.randn(1, 32, NQ, D).to(torch.float16).cuda()
    k = torch.randn(1, 32, NKV, D).to(torch.float16).cuda()
    v = torch.randn(1, 32, NKV, D).to(torch.float16).cuda()

    output = torch.empty_like(q)
    for head in range(q.size(1)):
        _q = q[:, head, :, :].unsqueeze(1)
        _k = k[:, head, :, :].unsqueeze(1)
        _v = v[:, head, :, :].unsqueeze(1)
        _ck = ck[:, head, :, :].unsqueeze(1)
        _cv = cv[:, head, :, :].unsqueeze(1)

        vertical_size, slash_size = min(NQ, max(N_VIDX, 30)), min(NQ, max(N_SIDX, 50))
        last_q = min(64, NQ // 2)
        qk = torch.einsum(f'bhmk, bhnk -> bhmn', _q[:, :, -last_q:, :], _k) / math.sqrt(D)
        qk[:, :, :, -last_q:] = torch.where(LAST_Q_MASK[..., -last_q:, -last_q:].to(q.device), qk[:, :, :, -last_q:], -torch.inf)
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        vertical = qk.sum(-2, keepdim=True)
        vertical[..., :30] = torch.inf
        vertical_topk = torch.topk(vertical, vertical_size, -1).indices

        def sum_all_diagonal_matrix(mat: torch.tensor):
            b, h, n, m = mat.shape
            zero_mat = torch.zeros((b, h, n, n)).to(mat.device)  # Zero matrix used for padding
            mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)  # pads the matrix on left and right
            mat_strided = mat_padded.as_strided((1, 1, n, n + m), (1, n * (2 * n + m), 2 * n + m + 1, 1))  # Change the strides
            sum_diags = torch.sum(mat_strided, 2)  # Sums the resulting matrix's columns
            return sum_diags[:, :, 1:]

        slash = sum_all_diagonal_matrix(qk)[..., :-last_q + 1]
        slash[..., -100:] = torch.inf
        slash_topk = slash
        slash = (NQ - 1) - torch.topk(slash, slash_size, -1).indices

        attn_output = vertical_slash_sparse_attention(_ck, _cv, cmask, _q, _k, _v, vertical_topk, slash)
        output[:, head:head + 1] = attn_output

    print(output.size())
