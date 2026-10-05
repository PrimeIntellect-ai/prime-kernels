"""SM90 (Hopper) backward of DeepSeek-V4.1 sparse attention, in CuTe DSL.

A persistent CTA per SM walks queries q = blockIdx.x, + gridDim.x, ...; for each query its 64
heads form the M = 64 side of every MMA, and its non-empty slots (compacted to the front of the
list by the preprocess) are walked in tiles of 64 gathered latent rows (K = V, 512 wide):

    S  = Q K^T, dP = dO K^T                      (64 heads x 64 slots, K = 512)
    P  = exp2(S * scale_log2 - L2), dS = P (dP - delta) * scale
    dQ += dS K                                   (64 heads x 512, K = 64 slots)
    dKV[slots] += P^T dO + dS^T Q                (64 slots x 512, K = 64 heads), scattered with
                                                 fp32 vector atomics into a column-permuted buffer

Q and dO (64 x 512 each) stay resident; with them, one KV tile and P / dS the CTA uses ~215 KB of
shared memory, so the KV tile is single-buffered and split into two 256-column halves, each
refilled as soon as its last reader is done:

    WG0: S, dP (lo half then hi half) -> softmax -> P, dS to smem -> dQ[:, :256] += dS K_lo (dS from
         registers) -> gather K_lo of the next tile -> the first `wg0_chunks` dKV chunks
    WG1: dQ[:, 256:] += dS K_hi -> gather K_hi of the next tile -> the remaining dKV chunks
         (64 columns each; rotating accumulators, so a chunk's atomics overlap the next MMAs)

The next tile may belong to the next query: its slot list is prefetched into a second buffer
during the current query's first tile, and the halves of its Q / dO are loaded by TMA as soon as
the current query's last dKV chunk using them is done, so query switches do not drain the SM.
The dKV atomics (2 KB per query-slot, ~21 GB at V4.1 shapes) bound this kernel: on their own
they take ~6 ms at the ~3.5 TB/s an H200's L2 sustains for fp32 vector reductions.
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup

from prime_kernels.dsa_sparse_attn_bwd import _cute_utils as cu

HEADS = 64
DIM = 512
HALF = 256
TILE_N = 64
CHUNK = 64  # dKV is computed and scattered 64 latent columns at a time
NUM_CHUNKS = DIM // CHUNK
NUM_ACC = 3  # dKV accumulators in rotation: a chunk's atomics have two MMAs' time to read their registers

# named barriers (0 is __syncthreads)
BAR_PDS_FULL = 1  # WG0 -> WG1: P / dS of the current tile are in smem
BAR_PDS_FREE = 2  # WG1 -> WG0: WG1 is done reading P / dS
BAR_QDO_FREE = 3  # WG0 -> WG1: WG0's dKV chunks of a query's last tile are done (Q / dO reusable)


class SparseAttnBwdSm90:
    def __init__(self, num_slots: int, wg0_chunks: int = 2):
        assert num_slots % TILE_N == 0
        assert 0 <= wg0_chunks <= NUM_CHUNKS // 2
        self.wg0_chunks = wg0_chunks
        self.num_slots = num_slots
        self.num_threads = 256

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (t, h, d) bf16
        mKV: cute.Tensor,  # (n, d) bf16
        mdO: cute.Tensor,  # (t, h, d) bf16
        mL2: cute.Tensor,  # (t, h) f32, log2-domain sink-aware LSE
        mDelta: cute.Tensor,  # (t, h) f32
        mIdx: cute.Tensor,  # (t, k) int32, non-empty slots first, then -1
        mCount: cute.Tensor,  # (t,) int32, non-empty slots per query
        mdQ: cute.Tensor,  # (t, h, d) bf16
        mdKV: cute.Tensor,  # (n, d) f32, column-permuted accumulator
        scale: Float32,
        num_ctas: Int32,
        stream,
    ):
        def assume_aligned(t):
            divby = 128 // t.element_type.width
            strides = [s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1]]
            return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=(*strides, t.stride[-1])))

        mQ, mKV, mdO, mdQ, mdKV, mIdx = [assume_aligned(t) for t in (mQ, mKV, mdO, mdQ, mdKV, mIdx)]
        bf16 = cutlass.BFloat16
        sQ_layout = cu.make_smem_layout(bf16, (HEADS, HALF), 2)
        sK_layout = cu.make_smem_layout(bf16, (TILE_N, HALF), 2)
        sPdS_layout = cu.make_smem_layout(bf16, (HEADS, TILE_N))
        chunk_layout = cu.make_smem_layout(bf16, (HEADS, CHUNK))

        # (h, d, t) views so a (64, 256) TMA box is one query's half
        mQ_t, mdO_t = [cute.make_tensor(m.iterator, cute.select(m.layout, mode=[1, 2, 0])) for m in (mQ, mdO)]
        half_layout = cute.select(sQ_layout, mode=[0, 1])
        tma_Q, tma_mQ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), mQ_t, half_layout, (HEADS, HALF))
        tma_dO, tma_mdO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mdO_t, half_layout, (HEADS, HALF)
        )
        self.half_bytes = HEADS * HALF * 2

        mma_SdP = cu.make_tiled_mma("K", "K", TILE_N)
        mma_dQ_rs = cu.make_tiled_mma("K", "MN", HALF, a_in_regs=True)
        mma_dQ_ss = cu.make_tiled_mma("K", "MN", HALF)
        mma_dKV = cu.make_tiled_mma("MN", "MN", CHUNK)

        @cute.struct
        class SharedStorage:
            # Q/dO halves (TMA), K halves (cp.async), slot list (cp.async)
            mbar: cute.struct.MemRange[cutlass.Int64, 6]
            sIdx: cute.struct.Align[cute.struct.MemRange[Int32, 2 * self.num_slots], 128]
            sQ: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout)], 1024]
            sdO: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sQ_layout)], 1024]
            sK: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sK_layout)], 1024]
            sP: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sPdS_layout)], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[bf16, cute.cosize(sPdS_layout)], 1024]

        self.shared_storage = SharedStorage
        scale_log2 = scale * math.log2(math.e)
        self.kernel(
            tma_mQ,
            tma_mdO,
            tma_Q,
            tma_dO,
            mKV,
            mL2,
            mDelta,
            mIdx,
            mCount,
            mdQ,
            mdKV,
            mma_SdP,
            mma_dQ_rs,
            mma_dQ_ss,
            mma_dKV,
            sQ_layout,
            sK_layout,
            sPdS_layout,
            chunk_layout,
            scale,
            scale_log2,
        ).launch(
            grid=[num_ctas, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mdO: cute.Tensor,
        tma_Q: cute.CopyAtom,
        tma_dO: cute.CopyAtom,
        mKV: cute.Tensor,
        mL2: cute.Tensor,
        mDelta: cute.Tensor,
        mIdx: cute.Tensor,
        mCount: cute.Tensor,
        mdQ: cute.Tensor,
        mdKV: cute.Tensor,
        mma_SdP: cute.TiledMma,
        mma_dQ_rs: cute.TiledMma,
        mma_dQ_ss: cute.TiledMma,
        mma_dKV: cute.TiledMma,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        chunk_layout: cute.ComposedLayout,
        scale: Float32,
        scale_log2: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        wg = cute.arch.make_warp_uniform(tidx // 128)
        wg_tidx = tidx % 128
        num_queries = mCount.shape[0]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mbar0 = storage.mbar.data_ptr()
        mbar_QdO = (mbar0, mbar0 + 1)
        mbar_K = (mbar0 + 2, mbar0 + 3)
        mbar_idx = mbar0 + 4
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sP = storage.sP.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sIdx = storage.sIdx.get_tensor(cute.make_layout((self.num_slots, 2)))

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_Q)
            cpasync.prefetch_descriptor(tma_dO)
            cute.arch.mbarrier_init(mbar_QdO[0], 1)
            cute.arch.mbarrier_init(mbar_QdO[1], 1)
            cute.arch.mbarrier_init(mbar_K[0], 128)
            cute.arch.mbarrier_init(mbar_K[1], 128)
            cute.arch.mbarrier_init(mbar_idx, 128)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()

        # first query with work: slot list (WG0), Q / dO (WG1), first KV tile (both halves)
        q0 = self.next_query(mCount, bid - grid, grid, num_queries)
        if q0 < num_queries:
            if wg == 0:
                self.load_slots(mIdx, sIdx, q0, 0, wg_tidx, mbar_idx)
            elif warp_idx == 4:
                for h in cutlass.range_constexpr(2):
                    self.load_QdO_half(mQ, mdO, sQ, sdO, tma_Q, tma_dO, q0, h, mbar_QdO[h])
            cute.arch.mbarrier_wait(mbar_idx, 0)
            self.gather_half(mKV, sK, sIdx, 0, 0, wg, wg_tidx, mbar0 + 2 + wg)

        if wg == 0:
            self.wg0_loop(
                mKV,
                mL2,
                mDelta,
                mIdx,
                mCount,
                mdQ,
                mdKV,
                sQ,
                sdO,
                sK,
                sP,
                sdS,
                sIdx,
                mbar_QdO,
                mbar_K,
                mbar_idx,
                mma_SdP,
                mma_dQ_rs,
                mma_dKV,
                chunk_layout,
                scale,
                scale_log2,
                bid,
                grid,
                num_queries,
                wg_tidx,
            )
        else:
            self.wg1_loop(
                mQ,
                mdO,
                tma_Q,
                tma_dO,
                mKV,
                mCount,
                mdQ,
                mdKV,
                sQ,
                sdO,
                sK,
                sP,
                sdS,
                sIdx,
                mbar_QdO,
                mbar_K,
                mbar_idx,
                mma_dQ_ss,
                mma_dKV,
                chunk_layout,
                bid,
                grid,
                num_queries,
                wg_tidx,
                warp_idx,
            )

    @cute.jit
    def next_query(self, mCount: cute.Tensor, q: Int32, grid: Int32, num_queries) -> Int32:
        """The CTA's next query after `q` that has at least one non-empty slot (num_queries if none)."""
        qn = q + grid
        while qn < num_queries and mCount[qn] == 0:
            qn += grid
        return cutlass.min(qn, num_queries)

    @cute.jit
    def load_slots(self, mIdx: cute.Tensor, sIdx: cute.Tensor, q: Int32, buf: Int32, wg_tidx: Int32, mbar):
        """cp.async query `q`'s slot list into buffer `buf`; arrives on `mbar` (128 arrivals)."""
        cp_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), Int32, num_bits_per_copy=128
        )
        cp_thr = cute.make_tiled_copy_tv(cp_atom, cute.make_layout((1,)), cute.make_layout((4,))).get_slice(0)
        num_vec = self.num_slots // 4
        g_vecs = cute.flat_divide(mIdx[q, None], (4,))
        s_vecs = cute.flat_divide(sIdx[None, buf], (4,))
        for i in cutlass.range_constexpr((num_vec + 127) // 128):
            v = i * 128 + wg_tidx
            if v < num_vec:
                cute.copy(cp_atom, cp_thr.partition_S(g_vecs[None, v]), cp_thr.partition_D(s_vecs[None, v]))
        cute.arch.cp_async_mbarrier_arrive_noinc(mbar)

    @cute.jit
    def load_QdO_half(self, mQ, mdO, sQ, sdO, tma_Q, tma_dO, q: Int32, h: cutlass.Constexpr[int], mbar):
        """One thread: TMA half `h` (256 columns) of query `q`'s Q and dO into smem."""
        gQ = cute.local_tile(mQ[None, None, q], (HEADS, HALF), (0, None))
        gdO = cute.local_tile(mdO[None, None, q], (HEADS, HALF), (0, None))
        tQs, tQg = cpasync.tma_partition(
            tma_Q, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2), cute.group_modes(gQ, 0, 2)
        )
        tdOs, tdOg = cpasync.tma_partition(
            tma_dO, 0, cute.make_layout(1), cute.group_modes(sdO, 0, 2), cute.group_modes(gdO, 0, 2)
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, 2 * self.half_bytes)
        cute.copy(tma_Q, tQg[None, h], tQs[None, h], tma_bar_ptr=mbar)
        cute.copy(tma_dO, tdOg[None, h], tdOs[None, h], tma_bar_ptr=mbar)

    @cute.jit
    def gather_half(
        self,
        mKV: cute.Tensor,
        sK: cute.Tensor,
        sIdx: cute.Tensor,
        tile: Int32,
        buf: Int32,
        half: Int32,
        wg_tidx: Int32,
        mbar,
    ):
        """cp.async the 256-column half `half` of the tile's 64 rows (empty slots read row 0, masked
        later); completion arrives on `mbar` (one arrival per thread of the warpgroup)."""
        cp_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), cutlass.BFloat16, num_bits_per_copy=128
        )
        cp_thr = cute.make_tiled_copy_tv(cp_atom, cute.make_layout((1,)), cute.make_layout((8,))).get_slice(0)
        # 32 threads cover one row's 512-byte half: thread -> 16-byte chunk, rows strided by 4
        col = wg_tidx % 32
        for i in cutlass.range_constexpr(TILE_N // 4):
            row = wg_tidx // 32 + 4 * i
            kv_row = cutlass.max(sIdx[tile * TILE_N + row, buf], Int32(0))
            g_chunk = cute.flat_divide(mKV[kv_row, None], (8,))[None, half * 32 + col]
            s_chunk = cute.flat_divide(sK[row, None, half], (8,))[None, col]
            cute.copy(cp_atom, cp_thr.partition_S(g_chunk), cp_thr.partition_D(s_chunk))
        cute.arch.cp_async_mbarrier_arrive_noinc(mbar)

    @cute.jit
    def gather_next(
        self,
        mKV,
        sK,
        sIdx,
        tile,
        num_tiles,
        buf,
        q_next,
        num_queries,
        nq,
        half: cutlass.Constexpr[int],
        wg_tidx,
        mbar_K,
        mbar_idx,
    ):
        """Gather this warpgroup's half of the tile after (query, tile): the next tile of this query,
        or the first tile of the next query once its slot list has landed."""
        if tile + 1 < num_tiles:
            self.gather_half(mKV, sK, sIdx, tile + 1, buf, half, wg_tidx, mbar_K)
        elif q_next < num_queries:
            cute.arch.mbarrier_wait(mbar_idx, (nq + 1) % 2)
            self.gather_half(mKV, sK, sIdx, 0, 1 - buf, half, wg_tidx, mbar_K)

    @cute.jit
    def wg0_loop(
        self,
        mKV,
        mL2,
        mDelta,
        mIdx,
        mCount,
        mdQ,
        mdKV,
        sQ,
        sdO,
        sK,
        sP,
        sdS,
        sIdx,
        mbar_QdO,
        mbar_K,
        mbar_idx,
        mma_SdP,
        mma_dQ_rs,
        mma_dKV,
        chunk_layout,
        scale,
        scale_log2,
        bid,
        grid,
        num_queries,
        wg_tidx,
    ):
        dkv = self.dkv_setup(mma_dKV, sP, sdS, wg_tidx)
        thr_SdP = mma_SdP.get_slice(wg_tidx)
        tSrQ = thr_SdP.make_fragment_A(thr_SdP.partition_A(sQ))
        tSrdO = thr_SdP.make_fragment_A(thr_SdP.partition_A(sdO))
        tSrK = thr_SdP.make_fragment_B(thr_SdP.partition_B(sK))
        tScS = cu.acc_mn_view(thr_SdP.partition_C(cute.make_identity_tensor((HEADS, TILE_N))))

        thr_dQ = mma_dQ_rs.get_slice(wg_tidx)
        tdQrKt = thr_dQ.make_fragment_B(thr_dQ.partition_B(cu.transpose_view(sK)))
        acc_dQ = cute.make_rmem_tensor(mma_dQ_rs.partition_shape_C((HEADS, HALF)), Float32)

        smem_store = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), cutlass.BFloat16
        )
        thr_store = cute.make_tiled_copy_C(smem_store, mma_SdP).get_slice(wg_tidx)
        tPsP = thr_store.partition_D(sP)
        tPsdS = thr_store.partition_D(sdS)
        row_l2 = cute.make_rmem_tensor((2,), Float32)
        row_delta = cute.make_rmem_tensor((2,), Float32)

        it = Int32(0)  # tiles done by this CTA (phase of the K barriers)
        nq = Int32(0)  # queries with work done by this CTA (phase of the Q/dO and slot barriers)
        q = bid
        while q < num_queries:
            num_tiles = (mCount[q] + TILE_N - 1) // TILE_N
            if num_tiles == 0:
                acc_dQ.fill(0.0)
                self.store_dQ(acc_dQ, thr_dQ, mdQ, q, 0)
                q += grid
            else:
                buf = nq % 2
                q_next = self.next_query(mCount, q, grid, num_queries)
                for r in cutlass.range_constexpr(2):
                    row_l2[r] = mL2[q, tScS[r, 0][0]]
                    row_delta[r] = mDelta[q, tScS[r, 0][0]]
                acc_dQ.fill(0.0)
                for tile in cutlass.range(num_tiles, unroll=1):
                    acc_S = cute.make_rmem_tensor(mma_SdP.partition_shape_C((HEADS, TILE_N)), Float32)
                    acc_dP = cute.make_rmem_tensor(mma_SdP.partition_shape_C((HEADS, TILE_N)), Float32)
                    for h in cutlass.range_constexpr(2):
                        cute.arch.mbarrier_wait(mbar_K[h], it % 2)
                        if tile == 0:
                            cute.arch.mbarrier_wait(mbar_QdO[h], nq % 2)
                        cute.arch.fence_view_async_shared()
                        cu.gemm(
                            mma_SdP, acc_S, tSrQ[None, None, None, h], tSrK[None, None, None, h], zero_init=(h == 0)
                        )
                        cu.gemm(
                            mma_SdP, acc_dP, tSrdO[None, None, None, h], tSrK[None, None, None, h], zero_init=(h == 0)
                        )
                    warpgroup.wait_group(0)

                    S = cu.acc_mn_view(acc_S)
                    dP = cu.acc_mn_view(acc_dP)
                    for c in cutlass.range_constexpr(cute.size(S, mode=[1])):
                        valid = sIdx[tile * TILE_N + tScS[0, c][1], buf] >= 0
                        for r in cutlass.range_constexpr(cute.size(S, mode=[0])):
                            p = cute.math.exp2(S[r, c] * scale_log2 - row_l2[r], fastmath=True)
                            p = cu.select_f32(valid, p, Float32(0.0))
                            S[r, c] = p
                            dP[r, c] = p * (dP[r, c] - row_delta[r]) * scale
                    rP = cu.to_bf16(acc_S)
                    rdS = cu.to_bf16(acc_dP)

                    if it > 0:
                        cute.arch.barrier(barrier_id=BAR_PDS_FREE, number_of_threads=256)
                    if tile == 0 and q_next < num_queries:
                        # WG1 is past the previous query: its slot buffer can take the next query's list
                        self.load_slots(mIdx, sIdx, q_next, 1 - buf, wg_tidx, mbar_idx)
                    cute.copy(thr_store, thr_store.retile(rP), tPsP)
                    cute.copy(thr_store, thr_store.retile(rdS), tPsdS)
                    cute.arch.fence_view_async_shared()
                    cute.arch.barrier_arrive(barrier_id=BAR_PDS_FULL, number_of_threads=256)

                    tdQrdS = cute.make_tensor(rdS.iterator, cu.acc_as_operand_a(rdS.layout))
                    cu.gemm(mma_dQ_rs, acc_dQ, tdQrdS, tdQrKt[None, None, None, 0], zero_init=False)
                    warpgroup.wait_group(0)
                    self.gather_next(
                        mKV, sK, sIdx, tile, num_tiles, buf, q_next, num_queries, nq, 0, wg_tidx, mbar_K[0], mbar_idx
                    )
                    if const_expr(self.wg0_chunks > 0):
                        # the first chunks of this tile's dKV, while the next tile is being gathered
                        self.dkv_rows(dkv, sIdx, tile, buf, mdKV, wg_tidx)
                        for c in cutlass.range_constexpr(self.wg0_chunks + 1):
                            if const_expr(c < self.wg0_chunks):
                                self.dkv_chunk(dkv, mma_dKV, sQ, sdO, chunk_layout, c)
                                if const_expr(c > 0):
                                    warpgroup.wait_group(1)
                            else:
                                warpgroup.wait_group(0)
                            if const_expr(c > 0):
                                self.scatter(dkv[3][(c - 1) % NUM_ACC], dkv[5], dkv[6], c - 1, mdKV, wg_tidx % 32)
                        if tile + 1 == num_tiles and q_next < num_queries:
                            cute.arch.barrier_arrive(barrier_id=BAR_QDO_FREE, number_of_threads=256)
                    it += 1
                self.store_dQ(acc_dQ, thr_dQ, mdQ, q, 0)
                self.zero_skipped_dQ(acc_dQ, thr_dQ, mdQ, q, q_next, grid, 0)
                nq += 1
                q = q_next
        if it > 0:
            cute.arch.barrier(barrier_id=BAR_PDS_FREE, number_of_threads=256)

    @cute.jit
    def dkv_setup(self, mma_dKV, sP, sdS, wg_tidx):
        thr_dKV = mma_dKV.get_slice(wg_tidx)
        tKVrPt = thr_dKV.make_fragment_A(thr_dKV.partition_A(cu.transpose_view(sP)))
        tKVrdSt = thr_dKV.make_fragment_A(thr_dKV.partition_A(cu.transpose_view(sdS)))
        tKVcKV = cu.acc_mn_view(thr_dKV.partition_C(cute.make_identity_tensor((TILE_N, CHUNK))))
        acc_shape = mma_dKV.partition_shape_C((TILE_N, CHUNK))
        accs = tuple(cute.make_rmem_tensor(acc_shape, Float32) for _ in range(NUM_ACC))
        rows = cute.make_rmem_tensor((2,), Int32)
        addrs = cute.make_rmem_tensor((4,), cutlass.Int64)
        preds = cute.make_rmem_tensor((4,), Int32)
        return (thr_dKV, tKVrPt, tKVrdSt, accs, tKVcKV, addrs, preds, rows)

    @cute.jit
    def dkv_rows(self, dkv, sIdx, tile, buf, mdKV, wg_tidx):
        thr_dKV, tKVrPt, tKVrdSt, accs, tKVcKV, addrs, preds, rows = dkv
        for r in cutlass.range_constexpr(2):
            rows[r] = sIdx[tile * TILE_N + tKVcKV[r, 0][0], buf]
        self.scatter_targets(rows, mdKV, wg_tidx % 32, addrs, preds)

    @cute.jit
    def dkv_chunk(self, dkv, mma_dKV, sQ, sdO, chunk_layout, c: cutlass.Constexpr[int]):
        """Issue (not wait) the MMA of dKV columns [64 c, 64 c + 64) into accumulator c % 2."""
        thr_dKV, tKVrPt, tKVrdSt, accs, tKVcKV, addrs, preds, rows = dkv
        half, cc = c // (NUM_CHUNKS // 2), c % (NUM_CHUNKS // 2)
        sdO_c = cute.make_tensor(sdO[None, None, half].iterator + cc * HEADS * CHUNK, chunk_layout.outer)
        sQ_c = cute.make_tensor(sQ[None, None, half].iterator + cc * HEADS * CHUNK, chunk_layout.outer)
        tKVrdOt = thr_dKV.make_fragment_B(thr_dKV.partition_B(cu.transpose_view(sdO_c)))
        tKVrQt = thr_dKV.make_fragment_B(thr_dKV.partition_B(cu.transpose_view(sQ_c)))
        cu.gemm2(mma_dKV, accs[c % NUM_ACC], tKVrPt, tKVrdOt, tKVrdSt, tKVrQt)

    @cute.jit
    def wg1_loop(
        self,
        mQ,
        mdO,
        tma_Q,
        tma_dO,
        mKV,
        mCount,
        mdQ,
        mdKV,
        sQ,
        sdO,
        sK,
        sP,
        sdS,
        sIdx,
        mbar_QdO,
        mbar_K,
        mbar_idx,
        mma_dQ_ss,
        mma_dKV,
        chunk_layout,
        bid,
        grid,
        num_queries,
        wg_tidx,
        warp_idx,
    ):
        thr_dQ = mma_dQ_ss.get_slice(wg_tidx)
        tdQrKt = thr_dQ.make_fragment_B(thr_dQ.partition_B(cu.transpose_view(sK)))
        tdQrdS = thr_dQ.make_fragment_A(thr_dQ.partition_A(sdS))
        acc_dQ = cute.make_rmem_tensor(mma_dQ_ss.partition_shape_C((HEADS, HALF)), Float32)

        dkv = self.dkv_setup(mma_dKV, sP, sdS, wg_tidx)
        chunks = list(range(self.wg0_chunks, NUM_CHUNKS))

        nq = Int32(0)
        q = bid
        while q < num_queries:
            num_tiles = (mCount[q] + TILE_N - 1) // TILE_N
            if num_tiles == 0:
                acc_dQ.fill(0.0)
                self.store_dQ(acc_dQ, thr_dQ, mdQ, q, 1)
                q += grid
            else:
                buf = nq % 2
                q_next = self.next_query(mCount, q, grid, num_queries)
                acc_dQ.fill(0.0)
                for tile in cutlass.range(num_tiles, unroll=1):
                    last = tile + 1 == num_tiles
                    cute.arch.barrier(barrier_id=BAR_PDS_FULL, number_of_threads=256)
                    self.dkv_rows(dkv, sIdx, tile, buf, mdKV, wg_tidx)
                    cu.gemm(mma_dQ_ss, acc_dQ, tdQrdS, tdQrKt[None, None, None, 1], zero_init=False)
                    reload = last and q_next < num_queries
                    for i in cutlass.range_constexpr(len(chunks) + 1):
                        if const_expr(i < len(chunks)):
                            self.dkv_chunk(dkv, mma_dKV, sQ, sdO, chunk_layout, chunks[i])
                        if const_expr(i == 0):
                            # dQ_hi (issued first) is done: the high half of the tile is free
                            warpgroup.wait_group(1)
                            self.gather_next(
                                mKV,
                                sK,
                                sIdx,
                                tile,
                                num_tiles,
                                buf,
                                q_next,
                                num_queries,
                                nq,
                                1,
                                wg_tidx,
                                mbar_K[1],
                                mbar_idx,
                            )
                        else:
                            if const_expr(i < len(chunks)):
                                warpgroup.wait_group(1)
                            else:
                                warpgroup.wait_group(0)
                                cute.arch.barrier_arrive(barrier_id=BAR_PDS_FREE, number_of_threads=256)
                        # once the last chunk reading a half of Q / dO is done, load the next query's
                        done = chunks[i - 1] if i > 0 else self.wg0_chunks - 1
                        if const_expr(done == NUM_CHUNKS // 2 - 1 and (i > 0 or self.wg0_chunks == NUM_CHUNKS // 2)):
                            if reload:
                                if const_expr(self.wg0_chunks > 0):
                                    cute.arch.barrier(barrier_id=BAR_QDO_FREE, number_of_threads=256)
                                if warp_idx == 4:
                                    self.load_QdO_half(mQ, mdO, sQ, sdO, tma_Q, tma_dO, q_next, 0, mbar_QdO[0])
                        if const_expr(i == len(chunks)):
                            if reload and warp_idx == 4:
                                self.load_QdO_half(mQ, mdO, sQ, sdO, tma_Q, tma_dO, q_next, 1, mbar_QdO[1])
                        if const_expr(i > 0):
                            self.scatter(
                                dkv[3][chunks[i - 1] % NUM_ACC], dkv[5], dkv[6], chunks[i - 1], mdKV, wg_tidx % 32
                            )
                self.store_dQ(acc_dQ, thr_dQ, mdQ, q, 1)
                self.zero_skipped_dQ(acc_dQ, thr_dQ, mdQ, q, q_next, grid, 1)
                nq += 1
                q = q_next

    @cute.jit
    def zero_skipped_dQ(self, acc_dQ, thr_mma, mdQ, q, q_next, grid, half: cutlass.Constexpr[int]):
        """dQ = 0 for the queries between q and q_next that `next_query` skipped (no non-empty slot)."""
        qq = q + grid
        if qq < q_next:
            acc_dQ.fill(0.0)
            while qq < q_next:
                self.store_dQ(acc_dQ, thr_mma, mdQ, qq, half)
                qq += grid

    @cute.jit
    def store_dQ(self, acc_dQ: cute.Tensor, thr_mma, mdQ: cute.Tensor, q: Int32, half: cutlass.Constexpr[int]):
        """Write this warpgroup's (64 x 256) half of the query's dQ straight from registers."""
        rdQ = cu.to_bf16(acc_dQ)
        gdQ = cute.local_tile(mdQ[q, None, None], (HEADS, HALF), (0, half))
        cute.autovec_copy(rdQ, thr_mma.partition_C(gdQ))

    @cute.jit
    def scatter_targets(
        self, rows: cute.Tensor, mdKV: cute.Tensor, lane: Int32, addrs: cute.Tensor, preds: cute.Tensor
    ):
        """Per tile: the two latent rows (a, b) each of this thread's accumulator rows feeds after the
        quad swap in `scatter`, as byte addresses (lane and half-group offsets folded in) and validity."""
        is_b = ((lane // 4) % 2) == 1
        base = mdKV.iterator.toint() + (lane % 4) * 16 + cutlass.select_(is_b, Int32(64), Int32(0))
        for r in cutlass.range_constexpr(2):
            own_row = rows[r]
            partner_row = cute.arch.shuffle_sync_bfly(own_row, offset=4)
            row_a = cutlass.select_(is_b, partner_row, own_row)
            row_b = cutlass.select_(is_b, own_row, partner_row)
            addrs[2 * r] = base + cutlass.Int64(cutlass.max(row_a, Int32(0))) * (DIM * 4)
            addrs[2 * r + 1] = base + cutlass.Int64(cutlass.max(row_b, Int32(0))) * (DIM * 4)
            preds[2 * r] = Int32(row_a >= 0)
            preds[2 * r + 1] = Int32(row_b >= 0)

    @cute.jit
    def scatter(
        self,
        acc: cute.Tensor,
        addrs: cute.Tensor,
        preds: cute.Tensor,
        chunk: cutlass.Constexpr[int],
        mdKV: cute.Tensor,
        lane: Int32,
    ):
        """Atomically add a (64 slots x 64 cols) accumulator chunk to the latent rows its slots name.

        A thread of quad i (lane // 4) holds rows i and i + 8 of its warp's 16, columns
        {2 rank, 2 rank + 1} + 8 j (rank = lane % 4). Columns 16 g + {2 rank, 2 rank + 1, 8 + 2 rank,
        9 + 2 rank} are stored as the 4 contiguous floats at 16 g + 4 rank of the row's chunk (undone by
        the postprocess). Quads 2p and 2p + 1 swap halves so that each atomic instruction covers 4 rows
        x 128 contiguous bytes (8 rows x 64 bytes runs ~20% slower in L2).
        """
        A = cu.acc_mn_view(acc)
        is_b = ((lane // 4) % 2) == 1
        for r in cutlass.range_constexpr(2):
            for gp in cutlass.range_constexpr(2):
                g0, g1 = 2 * gp, 2 * gp + 1
                recv = cute.make_rmem_tensor((4,), Float32)
                for e in cutlass.range_constexpr(4):
                    recv[e] = cute.arch.shuffle_sync_bfly(
                        cu.select_f32(is_b, A[r, 4 * g0 + e], A[r, 4 * g1 + e]), offset=4
                    )
                offset = chunk * CHUNK * 4 + gp * 128
                cu.red_add_v4_if(
                    preds[2 * r],
                    addrs[2 * r] + offset,
                    cu.select_f32(is_b, recv[0], A[r, 4 * g0 + 0]),
                    cu.select_f32(is_b, recv[1], A[r, 4 * g0 + 1]),
                    cu.select_f32(is_b, recv[2], A[r, 4 * g0 + 2]),
                    cu.select_f32(is_b, recv[3], A[r, 4 * g0 + 3]),
                )
                cu.red_add_v4_if(
                    preds[2 * r + 1],
                    addrs[2 * r + 1] + offset,
                    cu.select_f32(is_b, A[r, 4 * g1 + 0], recv[0]),
                    cu.select_f32(is_b, A[r, 4 * g1 + 1], recv[1]),
                    cu.select_f32(is_b, A[r, 4 * g1 + 2], recv[2]),
                    cu.select_f32(is_b, A[r, 4 * g1 + 3], recv[3]),
                )
