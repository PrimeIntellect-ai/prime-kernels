import pytest
import torch

from prime_kernels.indexed_attention.selection import compute_selection_scores, select_indexed_blocks

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def make_inputs(tokens, num_blocks, docs, ties):
    generator = torch.Generator(device="cuda").manual_seed(tokens + num_blocks)
    query = torch.randn(tokens, 4, 128, device="cuda", generator=generator).bfloat16()
    key = torch.randn(num_blocks, 128, device="cuda", generator=generator).bfloat16()
    if ties:
        key[torch.rand(num_blocks, device="cuda", generator=generator) < 0.5] = 0
        key[1::3] = key[0::3][: key[1::3].shape[0]]
    total_tokens = num_blocks * 4
    doc_bounds = torch.linspace(0, total_tokens, docs + 1, device="cuda").long() // 4 * 4
    positions = torch.arange(total_tokens - tokens, total_tokens, device="cuda")
    doc = torch.searchsorted(doc_bounds[1:], positions, right=True)
    block_starts = doc_bounds[doc] // 4
    block_ends = block_starts + (positions - doc_bounds[doc] + 1) // 4
    return query, key, block_starts.int(), block_ends.int()


def reference(query, key, block_starts, block_ends, topk):
    scores, visible = compute_selection_scores(query, key, block_starts, block_ends)
    selected = torch.full((query.shape[0], topk), key.shape[0], dtype=torch.int32, device="cuda")
    for row in range(query.shape[0]):
        count = int(visible[row])
        order = torch.sort(scores[row, :count], descending=True, stable=True).indices[: min(topk, count)]
        selected[row, : order.numel()] = order.sort().values + block_starts[row]
    return selected


@pytest.mark.parametrize(
    "tokens,num_blocks,docs,ties", [(2049, 1023, 2, False), (1500, 1021, 1, True), (2051, 1024, 5, True)]
)
@pytest.mark.parametrize("topk", [64, 512])
def test_select_indexed_blocks_takes_lowest_tied_blocks_in_order(tokens, num_blocks, docs, ties, topk):
    inputs = make_inputs(tokens, num_blocks, docs, ties)
    selected = select_indexed_blocks(*inputs, topk)
    assert torch.equal(selected, reference(*inputs, topk))
    assert torch.equal(select_indexed_blocks(*inputs, topk), selected)
