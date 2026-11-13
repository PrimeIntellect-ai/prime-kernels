"""Small CuTe DSL helpers for the SM90 sparse attention backward."""

import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32
from cutlass._mlir.dialects import arith, llvm
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import LayoutEnum


def make_smem_layout(dtype, shape, stage=None):
    """Row-major (K-major) smem tile with the widest swizzle its row allows, optionally staged."""
    atom = warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(LayoutEnum.ROW_MAJOR, dtype, shape[1]), dtype
    )
    if stage is None:
        return cute.tile_to_shape(atom, shape, order=(0, 1))
    return cute.tile_to_shape(atom, (*shape, stage), order=(0, 1, 2))


def make_tiled_mma(a_major, b_major, tiler_n, a_in_regs=False):
    mode = {"K": cute.nvgpu.OperandMajorMode.K, "MN": cute.nvgpu.OperandMajorMode.MN}
    return sm90_utils.make_trivial_tiled_mma(
        cutlass.BFloat16,
        cutlass.BFloat16,
        mode[a_major],
        mode[b_major],
        Float32,
        atom_layout_mnk=(1, 1, 1),
        tiler_mn=(64, tiler_n),
        a_source=warpgroup.OperandSource.RMEM if a_in_regs else warpgroup.OperandSource.SMEM,
    )


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Swap the first two modes of a smem tensor."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma, acc: cute.Tensor, tA: cute.Tensor, tB: cute.Tensor, zero_init: cutlass.Constexpr[bool]
):
    """Issue one warpgroup MMA over every k block of `tA`/`tB` and commit it (does not wait)."""
    warpgroup.fence()
    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tA.shape[2])):
        cute.gemm(atom, acc, tA[None, None, k], tB[None, None, k], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()


def acc_mn_view(acc: cute.Tensor) -> cute.Tensor:
    """View an SM90 accumulator ((2, 2, V), M, N) as ((2, M), (2, V, N)): rows x columns."""
    layout = cute.make_layout(acc.layout.shape)
    shape = ((layout.shape[0][1], layout.shape[1]), (layout.shape[0][0], layout.shape[0][2], layout.shape[2]))
    stride = ((layout.stride[0][1], layout.stride[1]), (layout.stride[0][0], layout.stride[0][2], layout.stride[2]))
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, cute.make_layout(shape, stride=stride)))


def acc_as_operand_a(acc_layout: cute.Layout) -> cute.Layout:
    """Reinterpret a 16-bit copy of an accumulator ((2, 2, N/8), M, N') as the A fragment of the next MMA."""
    div = cute.logical_divide(acc_layout, ((None, None, 2), None, None))
    return cute.make_layout(
        ((div.shape[0][0], div.shape[0][1], div.shape[0][2][0]), div.shape[1], (div.shape[0][2][1], div.shape[2])),
        stride=(
            (div.stride[0][0], div.stride[0][1], div.stride[0][2][0]),
            div.stride[1],
            (div.stride[0][2][1], div.stride[2]),
        ),
    )


@dsl_user_op
def cvt_bf16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def to_bf16(src: cute.Tensor) -> cute.Tensor:
    dst = cute.make_rmem_tensor(src.shape, cutlass.BFloat16)
    dst_i32 = cute.recast_tensor(dst, cutlass.Int32)
    for i in cutlass.range_constexpr(cute.size(dst_i32)):
        dst_i32[i] = cvt_bf16x2(src[2 * i], src[2 * i + 1])
    return dst


@dsl_user_op
def select_f32(pred, a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        arith.select(
            cutlass.Boolean(pred).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
        )
    )


@cute.jit
def gemm2(
    tiled_mma: cute.TiledMma, acc: cute.Tensor, tA0: cute.Tensor, tB0: cute.Tensor, tA1: cute.Tensor, tB1: cute.Tensor
):
    """acc = A0 B0 + A1 B1 as one committed warpgroup MMA group (does not wait)."""
    warpgroup.fence()
    atom = cute.make_mma_atom(tiled_mma.op)
    atom.set(warpgroup.Field.ACCUMULATE, False)
    for k in cutlass.range_constexpr(cute.size(tA0.shape[2])):
        cute.gemm(atom, acc, tA0[None, None, k], tB0[None, None, k], acc)
        atom.set(warpgroup.Field.ACCUMULATE, True)
    for k in cutlass.range_constexpr(cute.size(tA1.shape[2])):
        cute.gemm(atom, acc, tA1[None, None, k], tB1[None, None, k], acc)
    warpgroup.commit_group()


@dsl_user_op
def red_add_v4_if(pred, ptr_i64, a: Float32, b: Float32, c: Float32, d: Float32, *, loc=None, ip=None) -> None:
    """If `pred`: fire-and-forget float4 atomic add to a 16-byte aligned global address (no branch)."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(pred).ir_value(loc=loc, ip=ip),
            cutlass.Int64(ptr_i64).ir_value(loc=loc, ip=ip),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            Float32(c).ir_value(loc=loc, ip=ip),
            Float32(d).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .pred p;\nsetp.ne.b32 p, $0, 0;\n@p red.relaxed.gpu.global.add.v4.f32 [$1], {$2, $3, $4, $5};\n}",
        "r,l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
