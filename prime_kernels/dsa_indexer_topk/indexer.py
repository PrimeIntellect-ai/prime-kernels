import torch
import triton

from prime_kernels.dsa_indexer_topk.kernels import (
    SUPER,
    _gather_topk_kernel,
    _group_max_kernel,
    _group_topk_kernel,
)
from prime_kernels.dsa_indexer_topk.quant import per_token_quant_fp8

GROUP_MAX_BYTES = 1 << 29
GROUP_MAX_BLOCK_M = 64
GROUP_MAX_BLOCK_N = 128
GATHER_TILE = 128
SELECT_CHUNK = 1024
SELECT_CAP = 1024
# The persistent kernels are latency bound: capping registers fits 4 programs per SM.
MAX_REGISTERS = 128

_RESIDENT_PROGRAMS: dict[tuple, int] = {}


def unsupported_shape_reason(num_heads: int, head_dim: int, block_size: int) -> str | None:
    """Why `dsv41_index_topk` cannot run these shapes, or None."""
    if num_heads < 16 or num_heads & (num_heads - 1):
        return f"the number of index heads must be a power of two >= 16, got {num_heads}"
    if head_dim < 32 or head_dim & (head_dim - 1):
        return f"the index head dim must be a power of two >= 32, got {head_dim}"
    if block_size < 1 or block_size > 16 or block_size & (block_size - 1):
        return f"the candidate block size must be a power of two <= 16, got {block_size}"
    return None


def _plain_group(max_entries_per_doc: int, topk: int) -> int:
    """Group size when no candidate blocks are emitted, where it is free: larger groups shrink the
    per-query group selection, smaller ones the entries rescored (`topk` groups' worth)."""
    return min(8, max(2, triton.next_power_of_2(triton.cdiv(max_entries_per_doc, 8 * topk))))


def _launch_persistent(kernel, n_rows: int, device: torch.device, make_args, **meta) -> None:
    """Launch a persistent per-row kernel with as many programs as fit on the GPU at once.

    `make_args(n_programs)` returns the positional arguments, scratch sized for `n_programs`.
    """
    key = (kernel.fn.__name__, device.index, tuple(sorted(meta.items())))
    if key not in _RESIDENT_PROGRAMS:
        compiled = kernel.warmup(*make_args(1), grid=(1,), **meta)
        compiled._init_handles()
        props = torch.cuda.get_device_properties(device)
        regs_per_warp = -(-compiled.n_regs * 32 // 256) * 256
        by_regs = props.regs_per_multiprocessor // regs_per_warp // meta["num_warps"]
        by_smem = props.shared_memory_per_multiprocessor // (compiled.metadata.shared + 1024)
        by_threads = props.max_threads_per_multi_processor // (32 * meta["num_warps"])
        _RESIDENT_PROGRAMS[key] = props.multi_processor_count * max(1, min(by_regs, by_smem, by_threads))
    n_programs = min(n_rows, _RESIDENT_PROGRAMS[key])
    kernel[(n_programs,)](*make_args(n_programs), **meta)


def _group_max(q_fp8, k_fp8, k_scale, w, ks, ke, n_groups: int, group: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`(S_q, n_groups)` group maxima and `(S_q, n_groups / SUPER)` super-group maxima."""
    S_q, H, D = q_fp8.shape
    gm = torch.empty(S_q, n_groups, dtype=torch.float32, device=q_fp8.device)
    sgm = torch.empty(S_q, triton.cdiv(n_groups, SUPER.value), dtype=torch.float32, device=q_fp8.device)
    grid = (triton.cdiv(S_q, GROUP_MAX_BLOCK_M), triton.cdiv(n_groups * group, GROUP_MAX_BLOCK_N))
    _group_max_kernel[grid](
        q_fp8,
        k_fp8,
        k_scale,
        w,
        ks,
        ke,
        gm,
        sgm,
        S_q,
        k_fp8.shape[0],
        n_groups,
        sgm.shape[1],
        q_fp8.stride(0),
        q_fp8.stride(1),
        k_fp8.stride(0),
        w.stride(0),
        gm.stride(0),
        sgm.stride(0),
        H=H,
        D=D,
        GROUP=group,
        BLOCK_M=GROUP_MAX_BLOCK_M,
        BLOCK_N=GROUP_MAX_BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return gm, sgm


def _group_topk(gm, sgm, ks, ke, out, tau, k: int, group: int, pin_newest: bool) -> None:
    """Best `k` groups per query into `out`; with `tau`, also the rescoring threshold."""

    def make_args(n_programs):
        return (
            gm,
            sgm,
            ks,
            ke,
            torch.empty(n_programs, gm.shape[1], dtype=torch.float32, device=gm.device),
            torch.empty(n_programs, gm.shape[1], dtype=torch.int32, device=gm.device),
            out,
            out if tau is None else tau,
            gm.shape[0],
            gm.shape[1],
            sgm.shape[1],
            gm.stride(0),
            sgm.stride(0),
            out.stride(0),
        )

    _launch_persistent(
        _group_topk_kernel,
        gm.shape[0],
        gm.device,
        make_args,
        K=k,
        GROUP=group,
        PIN_NEWEST=pin_newest,
        WRITE_TAU=tau is not None,
        CHUNK=SELECT_CHUNK,
        CAP=max(SELECT_CAP, triton.next_power_of_2(k)),
        num_warps=4,
        maxnreg=MAX_REGISTERS,
    )


def _gather_topk(q_fp8, k_fp8, k_scale, w, ks, ke, blocks, tau, out, topk: int, group: int) -> None:
    """Best `topk` entries per query among its `blocks`; `tau` from `_group_topk` when they are its picks."""
    S_q, H, D = q_fp8.shape
    n_cols = blocks.shape[1] * group

    def make_args(n_programs):
        return (
            q_fp8,
            k_fp8,
            k_scale,
            w,
            ks,
            ke,
            blocks,
            blocks if tau is None else tau,
            torch.empty(n_programs, 2 * n_cols + blocks.shape[1], dtype=torch.float32, device=q_fp8.device),
            torch.empty(n_programs, n_cols, dtype=torch.int32, device=q_fp8.device),
            out,
            S_q,
            k_fp8.shape[0],
            blocks.shape[1],
            q_fp8.stride(0),
            q_fp8.stride(1),
            k_fp8.stride(0),
            w.stride(0),
            blocks.stride(0),
            out.stride(0),
        )

    _launch_persistent(
        _gather_topk_kernel,
        S_q,
        q_fp8.device,
        make_args,
        H=H,
        D=D,
        GROUP=group,
        TOPK=topk,
        TILE=GATHER_TILE,
        CHUNK=SELECT_CHUNK,
        CAP=max(SELECT_CAP, triton.next_power_of_2(topk)),
        KNOWN_TAU=tau is not None,
        SCORE_STAGES=2,
        maxnreg=MAX_REGISTERS,
        num_warps=4,
    )


@torch.library.custom_op("prime_kernels::dsv41_index_topk", mutates_args=())
def dsv41_index_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    candidates: torch.Tensor | None,
    emit_candidates: bool,
    max_entries_per_doc: int,
    block_size: int,
    topk_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V4.1 Lightning Indexer top-k without a `(queries, entries)` score matrix.

    Drop-in for prime-rl's `dsv41_index_topk` (same arguments, same results up to ties):
    `score[t, e] = sum_h relu(q[t, h] . k[e]) * w[t, h]` over e in `[ks[t], ke[t])`, computed on
    per-token FP8 (UE8M0-scaled) q and k with fp32 accumulation.

    Args:
        q: `(S_q, H, D)` bf16 index queries. k: `(S_k, D)` bf16 index keys, shared across heads.
        w: `(S_q, H)` per-head weights.
        ks, ke: `(S_q,)` int32, each query's readable entries `[ks, ke)`; `ks` is its document's first
            entry and `ke - ks <= max_entries_per_doc`.
        candidates: `(S_q, topk_blocks)` int32 doc-local block indices (-1 unused) restricting the
            ranked entries, or None.
        emit_candidates: also return each query's `topk_blocks` best doc-local blocks of `block_size`
            entries, ranked by their best entry, with the block of the query's newest entry pinned.

    Returns `(S_q, topk)` int64 entry indices (unordered, -1 where a query has fewer readable
    entries) and the `(S_q, topk_blocks)` int32 candidate blocks (an empty tensor unless
    `emit_candidates`).

    Without `candidates`, a dense GEMM keeps only the max of every doc-local group of entries
    (`block_size` entries when emitting candidates, else `_plain_group`; `(S_q, S_k / group)` fp32,
    processed in query chunks of about `GROUP_MAX_BYTES`), the top `topk` groups are selected from
    them, and only those groups' entries are rescored and ranked. With `candidates`, only the
    candidate entries are scored.
    """
    S_q, H, D = q.shape
    reason = unsupported_shape_reason(H, D, block_size)
    if reason is not None:
        raise ValueError(reason)
    device = q.device
    out = torch.empty((S_q, topk), dtype=torch.int64, device=device)
    cand_out = torch.empty((S_q, topk_blocks) if emit_candidates else (0, 0), dtype=torch.int32, device=device)
    if S_q == 0 or topk == 0:
        return out, cand_out

    q_fp8, q_scale = per_token_quant_fp8(q.contiguous())
    if k.shape[0] == 0:
        # Nothing is readable; a zero row keeps every (masked) key load in bounds.
        k = k.new_zeros(1, D)
    k_fp8, k_scale = per_token_quant_fp8(k.contiguous())
    w = (w.float() * q_scale).contiguous()
    ks = ks.to(torch.int32).contiguous()
    ke = ke.to(torch.int32).contiguous()
    torch._assert_async((ke - ks <= max_entries_per_doc).all(), "dsv41_index_topk: ke - ks > max_entries_per_doc")

    group = block_size if emit_candidates else _plain_group(max_entries_per_doc, topk)
    n_groups = triton.cdiv(max_entries_per_doc, group)
    best_groups = torch.empty((S_q, topk), dtype=torch.int32, device=device) if candidates is None else None
    tau = torch.empty(S_q, dtype=torch.float32, device=device) if candidates is None else None
    if best_groups is not None or emit_candidates:
        rows_per_chunk = max(1, GROUP_MAX_BYTES // (4 * max(n_groups, 1)))
        for r0 in range(0, S_q, rows_per_chunk):
            r1 = min(S_q, r0 + rows_per_chunk)
            gm, sgm = _group_max(q_fp8[r0:r1], k_fp8, k_scale, w[r0:r1], ks[r0:r1], ke[r0:r1], n_groups, group)
            if best_groups is not None:
                _group_topk(
                    gm, sgm, ks[r0:r1], ke[r0:r1], best_groups[r0:r1], tau[r0:r1], topk, group, pin_newest=False
                )
            if emit_candidates:
                _group_topk(
                    gm, sgm, ks[r0:r1], ke[r0:r1], cand_out[r0:r1], None, topk_blocks, block_size, pin_newest=True
                )
            del gm, sgm  # free this chunk's maxima before the next one is allocated
    blocks = best_groups if candidates is None else candidates.to(torch.int32).contiguous()
    _gather_topk(q_fp8, k_fp8, k_scale, w, ks, ke, blocks, tau, out, topk, group if candidates is None else block_size)
    return out, cand_out


@dsv41_index_topk.register_fake
def _dsv41_index_topk_fake(
    q, k, w, ks, ke, topk, candidates, emit_candidates, max_entries_per_doc, block_size, topk_blocks
):
    cand_shape = (q.shape[0], topk_blocks) if emit_candidates else (0, 0)
    return q.new_empty((q.shape[0], topk), dtype=torch.int64), q.new_empty(cand_shape, dtype=torch.int32)
