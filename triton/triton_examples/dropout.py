import tabulate
import torch

import triton
import triton.language as tl


@triton.jit
def _dropout(
    x_ptr,  # pointer to the input
    x_keep_ptr,  # pointer to a mask of 0s and 1s
    output_ptr,  # pointer to the output
    n_elements,  # number of elements in the `x` tensor
    p,  # probability that an element of `x` is changed to zero
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    # The line below is the crucial part, described in the paragraph above!
    output = tl.where(x_keep, x / (1 - p), 0.0)
    # Write-back output
    tl.store(output_ptr + offsets, output, mask=mask)


def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


# # Input tensor
# x = torch.randn(size=(10, )).cuda()
# # Dropout mask
# p = 0.5
# x_keep = (torch.rand(size=(10, )) > p).to(torch.int32).cuda()
# #
# output = dropout(x, x_keep=x_keep, p=p)
# print(tabulate.tabulate([
#     ["input"] + x.tolist(),
#     ["keep mask"] + x_keep.tolist(),
#     ["output"] + output.tolist(),
# ]))

@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # load data from x
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # randomly prune it
    random = tl.rand(seed, offsets)
    x_keep = random > p
    # write-back
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output

@triton.jit
def _seeded_matrix_dropout(
    x_ptr, x_stride_m, x_stride_n,
    output_ptr, output_stride,
    n_rows, n_cols, #M, N
    p,
    seed_ptr, seed_stride,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # compute memory offsets of elements handled by this instance
    pid_m, pid_n = tl.program_id(axis=0), tl.program_id(axis=1)
    row = pid_m * BLOCK_SIZE_M
    col = pid_n * BLOCK_SIZE_N
    num_elem = n_rows * n_cols
    for k in range(0, BLOCK_SIZE_M):
        block_start = row * x_stride_m + col * x_stride_n
        offsets = block_start + tl.arange(0, BLOCK_SIZE_N)

        # load data from x
        mask = offsets < num_elem # Does strides before mess this 
        x = tl.load(x_ptr + offsets, mask=mask)
        # randomly prune it
        seed = tl.load(seed_ptr + row)
        random = tl.rand(seed, offsets)
        x_keep = random > p
        # write-back
        output = tl.where(x_keep, x / (1 - p), 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)

        # Update for the next row
        row += 1



def seeded_matrix_dropout(x, p, seeds):
    # TODO: Check shape/len of seeds
    output = torch.empty_like(x)
    assert x.is_contiguous()
    M, N = x.shape
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']), triton.cdiv(N, meta['BLOCK_SIZE_N']))
    _seeded_matrix_dropout[grid](
        x, x.stride(0), x.stride(1), #
        output, output.stride(0),
        M, N,
        p,
        seeds, seeds.stride(0),
        BLOCK_SIZE_M=1,
        BLOCK_SIZE_N=1024,
    )
    return output


# x = torch.randn(size=(10, )).cuda()
# # Compare this to the baseline - dropout mask is never instantiated!
# output = seeded_dropout(x, p=0.5, seed=123)
# output2 = seeded_dropout(x, p=0.5, seed=123)
# output3 = seeded_dropout(x, p=0.5, seed=512)

# print(
#     tabulate.tabulate([
#         ["input"] + x.tolist(),
#         ["output (seed = 123)"] + output.tolist(),
#         ["output (seed = 123)"] + output2.tolist(),
#         ["output (seed = 512)"] + output3.tolist(),
#     ]))



x = torch.randn(size=(2, 5)).cuda()

# Compare this to the baseline - dropout mask is never instantiated!
output = seeded_matrix_dropout(x, p=0.5, seeds=torch.tensor([105, 21]).cuda())
output2 = seeded_matrix_dropout(x, p=0.5, seeds=torch.tensor([105, 21]).cuda())
output3 = seeded_matrix_dropout(x, p=0.5, seeds=torch.tensor([21, 105]).cuda())

print(
    tabulate.tabulate([
        ["input"] + x.tolist(),
        ["output (seed = 123)"] + output.tolist(),
        ["output (seed = 123)"] + output2.tolist(),
        ["output (seed = 512)"] + output3.tolist(),
    ]))