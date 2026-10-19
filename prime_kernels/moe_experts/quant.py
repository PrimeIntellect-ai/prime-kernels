"""Gluon kernels that quantize a 128 x 128 tile both per row and per column (transposed).

The per-column outputs are what the K-grouped weight-gradient GEMMs read: each column of the
tile becomes a contiguous run of 128 bytes. Tiles are staged in shared memory by TMA so they can
be read back in a row-major register layout (vectorized along columns, for the per-row groups)
and in a column-major one in which each thread owns a run of rows of one column (for the
per-column groups and their transposed stores), without per-element global address math.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, mbarrier, tma
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor

TILE = gl.constexpr(128)
NUM_WARPS = 8
# Row pass: 16 rows at a time, 8 consecutive columns per thread.
ROW_CHUNK = gl.constexpr(16)
ROW_LAYOUT = gl.constexpr(gl.BlockedLayout([1, 8], [2, 16], [NUM_WARPS, 1], [1, 0]))
# Column pass: 32 columns at a time; each thread owns 16 consecutive rows of one column and two
# neighbouring threads 32, so every transposed store fills whole 32-byte sectors.
COL_CHUNK = gl.constexpr(32)
COL_LAYOUT = gl.constexpr(gl.BlockedLayout([16, 1], [2, 16], [4, 2], [0, 1]))
SMEM_LAYOUT = gl.constexpr(gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2))


def descriptor(tensor):
    return TensorDescriptor.from_tensor(tensor, [128, 128], SMEM_LAYOUT.value)


@gluon.jit
def _tile(t, tile_expert_ptr, tile_row_start_ptr, tile_row_end_ptr, expert_row_start_ptr, counts_ptr):
    """Tile ``t``'s expert and source rows, where its rows sit in its expert's K-grouped block, and
    how many of them are head rows (see `fp8`)."""
    expert = gl.load(tile_expert_ptr + t)
    row_start = gl.load(tile_row_start_ptr + t)
    num_valid = gl.load(tile_row_end_ptr + t) - row_start
    expert_start = gl.load(expert_row_start_ptr + expert)
    local_tile = (row_start - expert_start) // 128
    group_start = (t - local_tile).to(gl.int64) * 128
    group_rows = gl.cdiv(gl.load(counts_ptr + expert).to(gl.int32), 128) * 128
    head = gl.where(local_tile == 0, gl.minimum((128 - expert_start % 128) % 128, num_valid), 0)
    return expert, row_start, num_valid, local_tile * 128, group_start, group_rows, head


@gluon.jit
def _store_rows(v, out_ptr, sf_ptr, first_row, rows, cols, out_cols, M, sf_row, mask=None):
    """``v`` [rows, 128] quantized per row into rows ``first_row + rows`` of an ``[M, out_cols]``
    FP8 tensor with MN-major ``[K / 128, M]`` scales; ``sf_row`` is this K group. Offsets inside
    the tile are 32-bit, only its base is 64-bit."""
    scale = gl.maximum(gl.max(gl.abs(v), axis=1), 1e-10) / 448.0
    q = gl.minimum(gl.maximum(v * (1.0 / scale)[:, None], -448.0), 448.0).to(gl.float8e4nv)
    first_row = first_row.to(gl.int64)
    out = out_ptr + first_row * out_cols + (rows[:, None] * out_cols + cols[None, :])
    sf = sf_ptr + (sf_row * M + first_row) + rows
    if mask is None:
        gl.store(out, q)
        gl.store(sf, scale)
    else:
        gl.store(out, q, mask=mask[:, None])
        gl.store(sf, scale, mask=mask)


@gluon.jit
def _store_cols(v, out_ptr, sf_ptr, t, rows, cols, out_cols, local_row, group_start, group_rows):
    """``v`` [128 rows, cols] quantized per column into the K-grouped layout: an expert's block is
    ``[out_cols, group_rows]`` at element ``group_start * out_cols``, scales ``[Mp / 128,
    out_cols]``. Rows past the expert's end must be zeros."""
    scale = gl.maximum(gl.max(gl.abs(v), axis=0), 1e-10) / 448.0
    q = gl.minimum(gl.maximum(v * (1.0 / scale)[None, :], -448.0), 448.0).to(gl.float8e4nv)
    base = out_ptr + group_start * out_cols + local_row
    gl.store(base + (cols[None, :] * group_rows + rows[:, None]), q)
    gl.store(sf_ptr + t.to(gl.int64) * out_cols + cols, scale)


@gluon.jit
def _load_tiles(desc, row, col0, col1, buf0, buf1):
    """TMA loads of the ``[TILE, TILE]`` tiles of ``desc`` at ``(row, col0)`` and ``(row, col1)``."""
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    mbarrier.expect(bar, 2 * TILE * TILE * 2)
    tma.async_copy_global_to_shared(desc, [row, col0], bar, buf0)
    tma.async_copy_global_to_shared(desc, [row, col1], bar, buf1)
    mbarrier.wait(bar, 0)


@gluon.jit
def swiglu_kernel(
    gate_up_desc,
    h_ptr,
    h_sf_ptr,
    head_ptr,
    head_sf_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_row_end_ptr,
    expert_row_start_ptr,
    counts_ptr,
    num_tiles_ptr,
    M,
    M_HEAD,
    inter,
    limit,
):
    """h = clamped_swiglu(gate, up) from tile ``t`` of the padded gate/up GEMM output, quantized per
    row into the token-major layout (and its head rows into the head operand)."""
    t = gl.program_id(0)
    c = gl.program_id(1)
    if t >= gl.load(num_tiles_ptr):
        return
    expert, row_start, num_valid, _, _, _, head = _tile(
        t, tile_expert_ptr, tile_row_start_ptr, tile_row_end_ptr, expert_row_start_ptr, counts_ptr
    )
    gate_s = gl.allocate_shared_memory(gl.bfloat16, [TILE, TILE], SMEM_LAYOUT)
    up_s = gl.allocate_shared_memory(gl.bfloat16, [TILE, TILE], SMEM_LAYOUT)
    _load_tiles(gate_up_desc, t * TILE, c * TILE, inter + c * TILE, gate_s, up_s)
    cols = c * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(0, ROW_LAYOUT))
    for r in gl.static_range(0, TILE, ROW_CHUNK):
        rows = r + gl.arange(0, ROW_CHUNK, layout=gl.SliceLayout(1, ROW_LAYOUT))
        gate = gate_s.slice(r, ROW_CHUNK, dim=0).load(ROW_LAYOUT).to(gl.float32)
        up = up_s.slice(r, ROW_CHUNK, dim=0).load(ROW_LAYOUT).to(gl.float32)
        gate_c = gl.minimum(gate, limit)
        up_c = gl.minimum(gl.maximum(up, -limit), limit)
        h = gate_c / (1.0 + gl.exp(-gate_c)) * up_c
        _store_rows(h, h_ptr, h_sf_ptr, row_start, rows, cols, inter, M, c, rows < num_valid)
        if r < head:
            _store_rows(h, head_ptr, head_sf_ptr, expert * TILE, rows, cols, inter, M_HEAD, c, rows < head)


@gluon.jit
def _dswiglu(dh, gate, up, limit):
    gate_c = gl.minimum(gate, limit)
    up_c = gl.minimum(gl.maximum(up, -limit), limit)
    sig = 1.0 / (1.0 + gl.exp(-gate_c))
    silu = gate_c * sig
    dgate = gl.where(gate <= limit, dh * up_c * (sig * (1.0 + gate_c * (1.0 - sig))), 0.0)
    dup = gl.where((up >= -limit) & (up <= limit), dh * silu, 0.0)
    return dgate, dup, silu * up_c


@gluon.jit
def swiglu_backward_kernel(
    dh_desc,
    gate_up_desc,
    dgu_ptr,
    dgu_sf_ptr,
    head_ptr,
    head_sf_ptr,
    dgu_t_ptr,
    dgu_t_sf_ptr,
    h_t_ptr,
    h_t_sf_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_row_end_ptr,
    expert_row_start_ptr,
    counts_ptr,
    num_tiles_ptr,
    M,
    M_HEAD,
    inter,
    limit,
):
    """dgate | dup from tile ``t`` of the padded dh and gate/up: per row into the token-major
    layout (and the head operand) for dx, per column into the K-grouped layout for the gate/up
    weight gradient; and h, per column, for the down weight gradient. The column pass recomputes
    the SwiGLU backward from the staged inputs. The padding rows of the tile must be zeros (they
    are: see `fp8.Layout`), which makes their dgate, dup and h zeros too."""
    t = gl.program_id(0)
    c = gl.program_id(1)
    if t >= gl.load(num_tiles_ptr):
        return
    expert, row_start, num_valid, local_row, group_start, group_rows, head = _tile(
        t, tile_expert_ptr, tile_row_start_ptr, tile_row_end_ptr, expert_row_start_ptr, counts_ptr
    )
    dh_s = gl.allocate_shared_memory(gl.bfloat16, [TILE, TILE], SMEM_LAYOUT)
    gate_s = gl.allocate_shared_memory(gl.bfloat16, [TILE, TILE], SMEM_LAYOUT)
    up_s = gl.allocate_shared_memory(gl.bfloat16, [TILE, TILE], SMEM_LAYOUT)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    mbarrier.expect(bar, 3 * TILE * TILE * 2)
    tma.async_copy_global_to_shared(dh_desc, [t * TILE, c * TILE], bar, dh_s)
    tma.async_copy_global_to_shared(gate_up_desc, [t * TILE, c * TILE], bar, gate_s)
    tma.async_copy_global_to_shared(gate_up_desc, [t * TILE, inter + c * TILE], bar, up_s)
    mbarrier.wait(bar, 0)
    num_k = inter // TILE

    cols = c * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(0, ROW_LAYOUT))
    for r in gl.static_range(0, TILE, ROW_CHUNK):
        rows = r + gl.arange(0, ROW_CHUNK, layout=gl.SliceLayout(1, ROW_LAYOUT))
        valid = rows < num_valid
        dh = dh_s.slice(r, ROW_CHUNK, dim=0).load(ROW_LAYOUT).to(gl.float32)
        gate = gate_s.slice(r, ROW_CHUNK, dim=0).load(ROW_LAYOUT).to(gl.float32)
        up = up_s.slice(r, ROW_CHUNK, dim=0).load(ROW_LAYOUT).to(gl.float32)
        dgate, dup, _ = _dswiglu(dh, gate, up, limit)
        _store_rows(dgate, dgu_ptr, dgu_sf_ptr, row_start, rows, cols, 2 * inter, M, c, valid)
        _store_rows(dup, dgu_ptr, dgu_sf_ptr, row_start, rows, cols + inter, 2 * inter, M, c + num_k, valid)
        if r < head:
            in_head = rows < head
            _store_rows(dgate, head_ptr, head_sf_ptr, expert * TILE, rows, cols, 2 * inter, M_HEAD, c, in_head)
            _store_rows(
                dup, head_ptr, head_sf_ptr, expert * TILE, rows, cols + inter, 2 * inter, M_HEAD, c + num_k, in_head
            )

    rows = gl.arange(0, TILE, layout=gl.SliceLayout(1, COL_LAYOUT))
    for cc in gl.static_range(0, TILE, COL_CHUNK):
        chunk_cols = c * TILE + cc + gl.arange(0, COL_CHUNK, layout=gl.SliceLayout(0, COL_LAYOUT))
        dh = dh_s.slice(cc, COL_CHUNK, dim=1).load(COL_LAYOUT).to(gl.float32)
        gate = gate_s.slice(cc, COL_CHUNK, dim=1).load(COL_LAYOUT).to(gl.float32)
        up = up_s.slice(cc, COL_CHUNK, dim=1).load(COL_LAYOUT).to(gl.float32)
        dgate, dup, h = _dswiglu(dh, gate, up, limit)
        _store_cols(dgate, dgu_t_ptr, dgu_t_sf_ptr, t, rows, chunk_cols, 2 * inter, local_row, group_start, group_rows)
        _store_cols(
            dup, dgu_t_ptr, dgu_t_sf_ptr, t, rows, chunk_cols + inter, 2 * inter, local_row, group_start, group_rows
        )
        _store_cols(h, h_t_ptr, h_t_sf_ptr, t, rows, chunk_cols, inter, local_row, group_start, group_rows)


@gluon.jit
def quantize_kernel(
    src_desc,
    q_ptr,
    q_sf_ptr,
    qt_ptr,
    qt_sf_ptr,
    tile_expert_ptr,
    tile_row_start_ptr,
    tile_row_end_ptr,
    expert_row_start_ptr,
    counts_ptr,
    num_tiles_ptr,
    Mp,
    C,
    COLUMNS: gl.constexpr,
):
    """Tile ``t`` of the token-major ``src`` per row into the padded layout (its padding rows as
    zeros) and, with ``COLUMNS``, per column into the K-grouped layout."""
    t = gl.program_id(0)
    c = gl.program_id(1)
    if t >= gl.load(num_tiles_ptr):
        return
    _, row_start, num_valid, local_row, group_start, group_rows, _ = _tile(
        t, tile_expert_ptr, tile_row_start_ptr, tile_row_end_ptr, expert_row_start_ptr, counts_ptr
    )
    buf = gl.allocate_shared_memory(gl.bfloat16, [TILE, TILE], SMEM_LAYOUT)
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    mbarrier.expect(bar, TILE * TILE * 2)
    tma.async_copy_global_to_shared(src_desc, [row_start, c * TILE], bar, buf)
    mbarrier.wait(bar, 0)
    if num_valid < TILE:
        # The tile's last rows belong to the next expert: zero them once, in shared memory.
        for r in gl.static_range(0, TILE, ROW_CHUNK):
            rows = r + gl.arange(0, ROW_CHUNK, layout=gl.SliceLayout(1, ROW_LAYOUT))
            chunk = buf.slice(r, ROW_CHUNK, dim=0)
            chunk.store(gl.where((rows < num_valid)[:, None], chunk.load(ROW_LAYOUT), 0.0))
        gl.barrier()

    cols = c * TILE + gl.arange(0, TILE, layout=gl.SliceLayout(0, ROW_LAYOUT))
    for r in gl.static_range(0, TILE, ROW_CHUNK):
        rows = r + gl.arange(0, ROW_CHUNK, layout=gl.SliceLayout(1, ROW_LAYOUT))
        v = buf.slice(r, ROW_CHUNK, dim=0).load(ROW_LAYOUT).to(gl.float32)
        _store_rows(v, q_ptr, q_sf_ptr, t * TILE, rows, cols, C, Mp, c)
    if COLUMNS:
        rows = gl.arange(0, TILE, layout=gl.SliceLayout(1, COL_LAYOUT))
        for cc in gl.static_range(0, TILE, COL_CHUNK):
            chunk_cols = c * TILE + cc + gl.arange(0, COL_CHUNK, layout=gl.SliceLayout(0, COL_LAYOUT))
            v = buf.slice(cc, COL_CHUNK, dim=1).load(COL_LAYOUT).to(gl.float32)
            _store_cols(v, qt_ptr, qt_sf_ptr, t, rows, chunk_cols, C, local_row, group_start, group_rows)
