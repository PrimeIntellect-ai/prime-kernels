import torch


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _quantize_activations_fake(
    matrix: torch.Tensor,
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, contraction_size = matrix.shape
    groups = offsets.shape[0]
    scale_columns = _round_up(contraction_size // 16, 4)
    padded_scale_rows = _round_up(rows + groups * 127, 128)
    return (
        matrix.new_empty((rows, contraction_size // 2), dtype=torch.uint8),
        matrix.new_empty(
            (padded_scale_rows, scale_columns),
            dtype=torch.float8_e4m3fn,
        ),
        matrix.new_empty((rows,), dtype=torch.float32),
    )


def _quantize_weights_fake(
    weight_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups, output_size, contraction_size = weight_rows.shape
    padded_output_size = _round_up(output_size, 128)
    scale_columns = _round_up(contraction_size // 16, 4)
    return (
        weight_rows.new_empty(
            (groups, output_size, contraction_size // 2),
            dtype=torch.uint8,
        ),
        weight_rows.new_empty(
            (groups, padded_output_size * scale_columns),
            dtype=torch.float8_e4m3fn,
        ),
        weight_rows.new_empty((groups,), dtype=torch.float32),
    )


def _dequantize_activations_fake(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    del block_scales, global_scales, offsets
    return packed.new_empty(
        (packed.shape[0], packed.shape[1] * 2),
        dtype=torch.bfloat16,
    )


def _dequantize_weights_fake(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
) -> torch.Tensor:
    del block_scales, global_scales
    return packed.new_empty(
        (packed.shape[0], packed.shape[1], packed.shape[2] * 2),
        dtype=torch.bfloat16,
    )


def _grouped_mm_fake(
    activations: torch.Tensor,
    weight: torch.Tensor,
    activation_block_scales: torch.Tensor,
    weight_block_scales: torch.Tensor,
    offsets: torch.Tensor,
    activation_token_scales: torch.Tensor,
    weight_expert_scales: torch.Tensor,
) -> torch.Tensor:
    del (
        activation_block_scales,
        weight_block_scales,
        offsets,
        activation_token_scales,
        weight_expert_scales,
    )
    return activations.new_empty(
        (activations.shape[0], weight.shape[-1]),
        dtype=torch.bfloat16,
    )


torch.library.register_fake("prime_rl::quantize_nvfp4_activations", _quantize_activations_fake)
_quantize_activations = torch.ops.prime_rl.quantize_nvfp4_activations.default

torch.library.register_fake("prime_rl::quantize_nvfp4_weights", _quantize_weights_fake)
_quantize_weights = torch.ops.prime_rl.quantize_nvfp4_weights.default

torch.library.register_fake("prime_rl::dequantize_nvfp4_activations", _dequantize_activations_fake)
_dequantize_activations = torch.ops.prime_rl.dequantize_nvfp4_activations.default

torch.library.register_fake("prime_rl::dequantize_nvfp4_weights", _dequantize_weights_fake)
_dequantize_weights = torch.ops.prime_rl.dequantize_nvfp4_weights.default

torch.library.register_fake("prime_rl::grouped_nvfp4_gemm", _grouped_mm_fake)
_grouped_mm = torch.ops.prime_rl.grouped_nvfp4_gemm.default
