#pragma once

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

/*
    This is a more generic communication <-> computation (comm/comp) overlap system for CUDA.
    The core model is:
        tile_pipeline_kernel has 3 components which can be "plugged in":
            Scheduler: RoundRobin, Ordered, Expert-grouped etc..
            Transport: PeerStore, NVSHMEM, IBGDA, etc..
            Compute: tcgen05 ffn, attn etc..


        Producer CTAs:
            scheduler.next_send_tile()
                -> transport.post(tile)
                -> transport.producer_progtress()
        Consumer CTAs:
            scheduler.next_compute_tile()
                -> transport.acquire(tile)
                -> compute.exec(tile)
    The pipeline itself doesnt know wnything about
        -> NVLink
        -> NVSHMEM
        -> InfiniBand or IBGDA
    Those are supplied as policies.


    !! Memory Ordering !!
        Transport::post(tile)
            ☝ Must not publish the file readiness until all writes of that tile are visible to the receiving GPU
        Transport::acquire(tile)
            ☝ Must not return until the writers are visible to consumer CTAs

        For example: PeerStoreTransport implements this using:
            -> Remove stores, threadfence_system() and system-scope atomic flagss

*/

namespace pi {
    struct cta_block_ctx final {
        void *dyn_smem;
        int block_id, thread_id;
        int warp_id, lane_id;
        int com_slot;
        int prod_slot;
        int num_prod_blocks;
        int num_com_blocks;
    };

    template <typename T>
    concept kern_pod_policy = std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>;

    template <typename T>
    concept tile_transport = kern_pod_policy<T> && requires(T &&transport, int slot, int64_t tile, cta_block_ctx &ctx) {
        { transport.prod_init(slot, ctx) } -> std::same_as<void>;
        { transport.send_valid(tile) } -> std::convertible_to<bool>;
        { transport.post(tile, slot, ctx) } -> std::same_as<void>;
        { transport.prod_progress(slot, ctx) } -> std::same_as<void>;
        { transport.prod_epilogue(slot, ctx) } -> std::same_as<void>;
        { transport.acquire(tile, ctx) } -> std::same_as<void>;
    };

    template <typename C>
    concept tile_compute = kern_pod_policy<C> && requires(C &&compute, int64_t tile, cta_block_ctx &ctx) {
        { compute.init(ctx) } -> std::same_as<void>;
        { compute.valid(tile) } -> std::convertible_to<bool>;
        { compute.exec(tile, ctx) } -> std::same_as<void>;
        { compute.epilogue(ctx) } -> std::same_as<void>;
    };

    template <typename S>
    concept tile_scheduler = requires(S sched, int slot, int nblocks) {
            typename S::ProdState;
            typename S::ComState;
            { sched.prod_init(slot, nblocks) } -> std::same_as<typename S::ProdState>;
            { sched.com_init(slot, nblocks) } -> std::same_as<typename S::ComState>;
        } && requires(S sched, typename S::ProdState &prod_state, typename S::ComState &com_state) {
            { sched.next_send_tile(prod_state) } -> std::convertible_to<int64_t>;
            { sched.next_compute_tile(com_state) } -> std::convertible_to<int64_t>;
    };

    template <typename Transport, typename Compute, typename Scheduler>
        requires tile_transport<Transport> && tile_compute<Compute> && tile_scheduler<Scheduler>
    __global__ void tile_pipeline_kernel_hull(
        const __grid_constant__ Transport transport,
        const __grid_constant__ Compute compute,
        const __grid_constant__ Scheduler sched,
        int num_prod_blocks
    ) {
        extern __shared__ __align__(1024) uint8_t tilepipe_dynamic_smem[];
        int bid = static_cast<int>(blockIdx.x);
        int tid = static_cast<int>(threadIdx.x);
        int wid = tid>>5;
        int lid = 31&tid;
        int num_com_blocks = static_cast<int>(gridDim.x)-num_prod_blocks;
        if (bid < num_prod_blocks) { // Split: Producer CTA path
            cta_block_ctx ctx {
                tilepipe_dynamic_smem,
                bid,
                tid,
                wid,
                lid,
                -1,
                bid,
                num_prod_blocks,
                num_com_blocks
            };
            int prod_slot = bid;
            transport.prod_init(prod_slot, ctx);
            auto sched_state = sched.prod_init(prod_slot, num_prod_blocks);
            for (;;) {
                int64_t tile = sched.next_send_tile(sched_state);
                if (tile < 0) break;
                if (!transport.send_valid(tile)) continue;
                transport.post(tile, prod_slot, ctx);
                transport.prod_progress(prod_slot, ctx);
            }
            transport.prod_epilogue(prod_slot, ctx);
            return;
        }
        // Split: Consumer CTAs
        if (num_com_blocks <= 0) return;
        int com_slot = bid-num_prod_blocks;
        cta_block_ctx ctx {
            tilepipe_dynamic_smem,
            bid,
            tid,
            wid,
            lid,
            com_slot,
            -1,
            num_prod_blocks,
            num_com_blocks
        };
        compute.init(ctx);
        auto sched_state = sched.com_init(com_slot, num_com_blocks);
        for (;;) {
            int64_t tile = sched.next_compute_tile(sched_state);
            if (tile < 0) break;
            if (!compute.valid(tile)) continue;
            transport.acquire(tile, ctx);
            compute.exec(tile, ctx);
        }
        compute.epilogue(ctx);
    }

    template <typename Transport, typename Compute, typename Scheduler>
        requires tile_transport<Transport> && tile_compute<Compute> && tile_scheduler<Scheduler>
    [[nodiscard]] cudaError_t launch_tile_pipeline(
        Transport transport,
        Compute compute,
        Scheduler sched,
        int num_prod_blocks,
        int num_com_blocks,
        int threads_total,
        size_t dyn_smem_nb,
        cudaStream_t stream
    ) {
        int blocks_total = num_prod_blocks+num_com_blocks;
        if (__builtin_expect(blocks_total<=0, 0)) return cudaSuccess;
        if (__builtin_expect(threads_total<=0, 0)) return cudaErrorInvalidConfiguration;
        tile_pipeline_kernel_hull<Transport, Compute, Scheduler>
            <<<blocks_total, threads_total, dyn_smem_nb, stream>>>(transport, compute, sched, num_prod_blocks);
        return cudaGetLastError();
    }
}
