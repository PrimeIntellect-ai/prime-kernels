"""SM90 per-token (1 x 128) and transposed per-token (128 x 1) e4m3 casts of a bf16 matrix in one pass,
optionally after the inverse RoPE of each head's trailing 64 channels (the query gradient of DeepSeek-V4.1).

One CTA (256 threads) per [128 tokens x 128 channels] tile:
  1. each half-warp loads one token row of the tile (16 B per thread), rotates it if the tile holds RoPE
     channels, reduces the row's amax over its 16 lanes, writes the row's e4m3 values and scale, and parks
     the (rotated) bf16 row in shared memory;
  2. each thread takes half of one channel column, the two halves combine their amax through shared
     memory, and the column's e4m3 values go to a transposed shared tile;
  3. the transposed tile is written out, 128 contiguous bytes per channel.
The arithmetic is that of prime-rl's Triton kernels (`_triton_rope_inplace`, `_fp8_scale`, `_fp8_quantize`)
instruction for instruction, so the results are bitwise equal.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from prime_kernels.dsa_sparse_attn_bwd._cute_utils import cvt_bf16x2

BLOCK = 128
ROPE = 64
T_PITCH = BLOCK + 4  # bytes per channel row of the transposed tile (conflict-free 4-byte stores)


@dsl_user_op
def ld_global_v4(ptr, *, loc=None, ip=None):
    r = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int64(ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4))


def _asm(ret, asm, cons, args, side=False, loc=None, ip=None):
    return llvm.inline_asm(
        ret, [a.ir_value(loc=loc, ip=ip) for a in args], asm, cons, has_side_effects=side, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def ld_global_v4_f32(ptr, *, loc=None, ip=None):
    r = _asm(llvm.StructType.get_literal([T.f32()] * 4), "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
             "=f,=f,=f,=f,l", [cutlass.Int64(ptr)], loc=loc, ip=ip)
    return tuple(Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_shared_bf16x2_f32(addr, *, loc=None, ip=None):
    """Two adjacent bf16 from shared memory, as fp32 (low address first)."""
    r = _asm(llvm.StructType.get_literal([T.f32()] * 2),
             "{\n.reg .b32 w;\nld.shared.b32 w, [$2];\nshl.b32 $0, w, 16;\nand.b32 $1, w, 0xffff0000;\n}",
             "=f,=f,r", [Int32(addr)], side=True, loc=loc, ip=ip)
    return tuple(Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def bf16_lo(v, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), "shl.b32 $0, $1, 16;", "=f,r", [Int32(v)], loc=loc, ip=ip))


@dsl_user_op
def bf16_hi(v, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), "and.b32 $0, $1, 0xffff0000;", "=f,r", [Int32(v)], loc=loc, ip=ip))


@dsl_user_op
def st_shared_v4_b32(addr, a, b, c, d, *, loc=None, ip=None) -> None:
    _asm(None, "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", [Int32(addr), Int32(a), Int32(b), Int32(c),
         Int32(d)], side=True, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_u16_f32(addr, *, loc=None, ip=None) -> Float32:
    """A bf16 from shared memory, as fp32."""
    return Float32(_asm(T.f32(), "{\n.reg .b16 h;\nld.shared.b16 h, [$1];\ncvt.f32.bf16 $0, h;\n}", "=f,r",
                        [Int32(addr)], side=True, loc=loc, ip=ip))


@dsl_user_op
def st_shared_b32(addr, v, *, loc=None, ip=None) -> None:
    _asm(None, "st.shared.b32 [$0], $1;", "r,r", [Int32(addr), Int32(v)], side=True, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_b32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), "ld.shared.b32 $0, [$1];", "=r,r", [Int32(addr)], side=True, loc=loc, ip=ip))


@dsl_user_op
def st_shared_f32(addr, v, *, loc=None, ip=None) -> None:
    _asm(None, "st.shared.f32 [$0], $1;", "r,f", [Int32(addr), Float32(v)], side=True, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(_asm(T.f32(), "ld.shared.f32 $0, [$1];", "=f,r", [Int32(addr)], side=True, loc=loc, ip=ip))


@dsl_user_op
def st_global_v2(ptr, a, b, *, loc=None, ip=None) -> None:
    _asm(None, "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", [cutlass.Int64(ptr), Int32(a), Int32(b)], side=True,
         loc=loc, ip=ip)


@dsl_user_op
def st_global_b32(ptr, a, *, loc=None, ip=None) -> None:
    _asm(None, "st.global.b32 [$0], $1;", "l,r", [cutlass.Int64(ptr), Int32(a)], side=True, loc=loc, ip=ip)


def _f32_asm(asm: str, *args, loc=None, ip=None) -> Float32:
    constraints = "=f," + ",".join("f" for _ in args)
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip) for a in args],
            asm,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def round_bf16(a: Float32, *, loc=None, ip=None) -> Float32:
    """`a` rounded to bf16 (round to nearest even), as an fp32 value."""
    return _f32_asm("{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.f32.bf16 $0, h;\n}", a, loc=loc, ip=ip)


@dsl_user_op
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_asm("fma.rn.f32 $0, $1, $2, $3;", a, b, c, loc=loc, ip=ip)


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_asm("mul.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_asm("div.rn.f32 $0, $1, $2;", a, b, loc=loc, ip=ip)


@dsl_user_op
def max_abs(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """max(a, |b|)."""
    return _f32_asm("{\n.reg .f32 t;\nabs.f32 t, $2;\nmax.f32 $0, $1, t;\n}", a, b, loc=loc, ip=ip)


@dsl_user_op
def quantize_e4m3x2(lo: Float32, hi: Float32, rcp: Float32, scale: Float32, *, loc=None, ip=None) -> cutlass.Int32:
    """Two values / `scale` rounded to e4m3 (satfinite), as prime-rl's `_fp8_quantize` computes them
    (q = x * rcp, one Markstein correction, q kept where it is +-0); `lo` in the low byte."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (lo, hi, rcp, scale)],
            "{\n.reg .f32 q0, q1, r0, r1, y0, y1;\n.reg .pred z0, z1;\n.reg .b16 h;\n"
            "mul.rn.f32 q0, $1, $3;\nmul.rn.f32 q1, $2, $3;\n"
            "neg.f32 r0, q0;\nneg.f32 r1, q1;\n"
            "fma.rn.f32 r0, r0, $4, $1;\nfma.rn.f32 r1, r1, $4, $2;\n"
            "fma.rn.f32 y0, r0, $3, q0;\nfma.rn.f32 y1, r1, $3, q1;\n"
            "setp.eq.f32 z0, q0, 0f00000000;\nsetp.eq.f32 z1, q1, 0f00000000;\n"
            "selp.f32 y0, q0, y0, z0;\nselp.f32 y1, q1, y1, z1;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h, y1, y0;\ncvt.u32.u16 $0, h;\n}",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def neg_triton(a: Float32, *, loc=None, ip=None) -> Float32:
    """Triton's float negation, 0 - a (so -(+0) is +0)."""
    return _f32_asm("sub.rn.f32 $0, 0f00000000, $1;", a, loc=loc, ip=ip)


class DualCastSm90:
    def __init__(self, rope: bool, head_dim: int):
        assert head_dim % BLOCK == 0
        self.rope = rope
        self.head_dim = head_dim

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (rows, cols) bf16 (as int16)
        mCS: cute.Tensor,  # (positions, 64) f32 [cos | sin]
        mPos: cute.Tensor,  # (rows,) int32
        mOut: cute.Tensor,  # (rows, cols) uint8
        mSF: cute.Tensor,  # (cols / 128, >= rows) f32
        mOutT: cute.Tensor,  # (cols, padded rows) uint8
        mSFT: cute.Tensor,  # (padded rows / 128, cols) f32
        rows: Int32,
        stream,
    ):
        cols = mX.shape[1]
        prows = mOutT.shape[1]

        @cute.struct
        class Smem:
            tile: cute.struct.Align[cute.struct.MemRange[cutlass.Int16, BLOCK * BLOCK], 128]
            red: cute.struct.Align[cute.struct.MemRange[Float32, 4 * BLOCK], 128]

        self.smem = Smem
        self.kernel(mX, mCS, mPos, mOut, mSF, mOutT, mSFT, rows).launch(
            grid=[cols // BLOCK, prows // BLOCK, 1], block=[256, 1, 1], smem=Smem.size_in_bytes(), stream=stream
        )

    @cute.kernel
    def kernel(self, mX, mCS, mPos, mOut, mSF, mOutT, mSFT, rows: Int32):
        tid, _, _ = cute.arch.thread_idx()
        bk, bm, _ = cute.arch.block_idx()
        cols = mX.shape[1]
        prows = mOutT.shape[1]
        storage = cutlass.utils.SmemAllocator().allocate(self.smem)
        s_tile = storage.tile.data_ptr().toint()
        s_t = s_tile  # the transposed tile reuses the bf16 tile once every column is in registers
        s_red = storage.red.data_ptr().toint()
        x_base = mX.iterator.toint()
        out_base = mOut.iterator.toint()
        outt_base = mOutT.iterator.toint()

        c16 = tid % 16
        col0 = bk * BLOCK + 8 * c16
        rope_tile = (bk + 1) * BLOCK % self.head_dim == 0
        vals = cute.make_rmem_tensor((8,), Float32)
        bits = cute.make_rmem_tensor((4,), Int32)
        raw = cute.make_rmem_tensor((BLOCK // 16 * 4,), Int32)
        # 1. rows: load (all eight rows of this thread first), rotate, per-token cast, park in shared memory
        for i in cutlass.range_constexpr(BLOCK // 16):
            grow = bm * BLOCK + tid // 16 + 16 * i
            v = ld_global_v4(x_base + (cutlass.Int64(cutlass.min(grow, rows - 1)) * cols + col0) * 2)
            for e in cutlass.range_constexpr(4):
                raw[4 * i + e] = cutlass.select_(grow < rows, v[e], Int32(0))

        for i in cutlass.range_constexpr(BLOCK // 16):
            row = tid // 16 + 16 * i
            grow = bm * BLOCK + row
            for e in cutlass.range_constexpr(4):
                bits[e] = raw[4 * i + e]
            for e in cutlass.range_constexpr(4):
                vals[2 * e] = bf16_lo(bits[e])
                vals[2 * e + 1] = bf16_hi(bits[e])
            if const_expr(self.rope):
                if rope_tile and c16 >= (BLOCK - ROPE) // 8 and grow < rows:
                    cs_ptr = mCS.iterator.toint() + (cutlass.Int64(mPos[grow]) * ROPE + 4 * (c16 - (BLOCK - ROPE) // 8)) * 4
                    coss = ld_global_v4_f32(cs_ptr)
                    sins = ld_global_v4_f32(cs_ptr + ROPE // 2 * 4)
                    for e in cutlass.range_constexpr(4):
                        cs = coss[e]
                        sn = neg_triton(sins[e])
                        x1 = vals[2 * e]
                        x2 = vals[2 * e + 1]
                        y1 = round_bf16(fma_rn(x1, cs, neg_triton(mul_rn(x2, sn))))
                        y2 = round_bf16(fma_rn(x1, sn, mul_rn(x2, cs)))
                        vals[2 * e] = y1
                        vals[2 * e + 1] = y2
                        bits[e] = cvt_bf16x2(y1, y2)
            amax = Float32(0.0)
            for e in cutlass.range_constexpr(8):
                amax = max_abs(amax, vals[e])
            for k in cutlass.range_constexpr(4):
                amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=8 >> k))
            sc = div_rn(cute.arch.fmax(amax, Float32(1e-10)), Float32(448.0))
            rcp = div_rn(Float32(1.0), sc)
            q01 = quantize_e4m3x2(vals[0], vals[1], rcp, sc)
            q23 = quantize_e4m3x2(vals[2], vals[3], rcp, sc)
            q45 = quantize_e4m3x2(vals[4], vals[5], rcp, sc)
            q67 = quantize_e4m3x2(vals[6], vals[7], rcp, sc)
            if grow < rows:
                st_global_v2(out_base + cutlass.Int64(grow) * cols + col0, q01 | (q23 << 16), q45 | (q67 << 16))
                if c16 == 0:
                    mSF[bk, grow] = sc
            st_shared_v4_b32(s_tile + (row * BLOCK + 8 * c16) * 2, bits[0], bits[1], bits[2], bits[3])
        cute.arch.sync_threads()
        # 2. columns: thread = (column pair cp, token quarter tq); amax over the quarter, combined through shared
        # memory, then the transposed cast of both columns into shared memory
        cp = tid % (BLOCK // 2)
        tq = tid // (BLOCK // 2)
        lo = cute.make_rmem_tensor((BLOCK // 4,), Float32)
        hi = cute.make_rmem_tensor((BLOCK // 4,), Float32)
        amax_lo = Float32(0.0)
        amax_hi = Float32(0.0)
        for k in cutlass.range_constexpr(BLOCK // 4):
            v = ld_shared_bf16x2_f32(s_tile + ((tq * (BLOCK // 4) + k) * BLOCK + 2 * cp) * 2)
            lo[k] = v[0]
            hi[k] = v[1]
            amax_lo = max_abs(amax_lo, v[0])
            amax_hi = max_abs(amax_hi, v[1])
        st_shared_f32(s_red + (tq * BLOCK + 2 * cp) * 4, amax_lo)
        st_shared_f32(s_red + (tq * BLOCK + 2 * cp + 1) * 4, amax_hi)
        cute.arch.sync_threads()
        for h in cutlass.range_constexpr(2):
            c = 2 * cp + h
            amax = ld_shared_f32(s_red + c * 4)
            for j in cutlass.range_constexpr(1, 4):
                amax = cute.arch.fmax(amax, ld_shared_f32(s_red + (j * BLOCK + c) * 4))
            sc = div_rn(cute.arch.fmax(amax, Float32(1e-10)), Float32(448.0))
            rcp = div_rn(Float32(1.0), sc)
            if tq == 0:
                mSFT[bm, bk * BLOCK + c] = sc
            vv = lo if h == 0 else hi
            for k4 in cutlass.range_constexpr(BLOCK // 16):
                w = quantize_e4m3x2(vv[4 * k4], vv[4 * k4 + 1], rcp, sc) | (
                    quantize_e4m3x2(vv[4 * k4 + 2], vv[4 * k4 + 3], rcp, sc) << 16
                )
                st_shared_b32(s_t + c * T_PITCH + tq * (BLOCK // 4) + 4 * k4, w)
        cute.arch.sync_threads()
        # 3. write the transposed tile: channel row r, 32 words of 4 tokens
        lane = tid % 32
        warp = tid // 32
        for k in cutlass.range_constexpr(BLOCK // 8):
            r = warp + 8 * k
            w = ld_shared_b32(s_t + r * T_PITCH + 4 * lane)
            st_global_b32(outt_base + cutlass.Int64(bk * BLOCK + r) * prows + bm * BLOCK + 4 * lane, w)
