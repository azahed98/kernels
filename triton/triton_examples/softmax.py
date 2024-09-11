# Following SOTA Deep Learning Tutorials on YT
# https://www.youtube.com/watch?v=gyKBN1rnefI
# Also used the fused somftamx from the tutorials
# https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html#sphx-glr-getting-started-tutorials-02-fused-softmax-py

import torch
import triton

import triton.language as tl
import torch.nn.functional as F

from torch.profiler import profile, record_function, ProfilerActivity
from triton.runtime import driver

def softmax_pt(sample):
    return F.softmax(sample, dim=1)

def softmax_naive(x):
    x_max = x.max(dim=-1)[0]
    safe_x = x - x_max[..., None]
    exp_safe_x = torch.exp(safe_x)
    return exp_safe_x / torch.sum(exp_safe_x, dim=-1)[...,None]

@triton.jit
def _softmax_fwd_kernel(
    output_ptr,
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,
):
    ## Setup input ptrs
    
    # Row of our current program
    row_index = tl.program_id(0)
    
    # Ptr for the current program's row
    row_start_ptr = input_ptr + (row_index * stride_input_row)
    
    # How the offsets of each column in the row
    col_offsets = tl.arange(0, block_size)
    # Determine which offsets are actually valid
    row_mask = col_offsets < num_cols

    # Get the absolute pointers of each row elem
    input_pointers = row_start_ptr + col_offsets

    # Move From HBM to SRAM
    # other is what to fill the invalid data with
    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    # softmax calc
    safe_row = row - tl.max(row, axis=0)
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    # Need to write to output buffer
    output_row_ptr = output_ptr + (row_index * stride_output_row)
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)

def softmax_triton(x):
    """ Triton softmax forward pass """
    rows, cols = x.shape
    assert x.dim() == 2, f"only accepts 2D tensors for now"

    # Parallelize along the rows, so each col is a chunk
    # Last chunk may have extra allocated, need to mask later
    block_size = triton.next_power_of_2(cols)

    # TODO: "Thread density"
    num_warps = 4
    if block_size > 2047:
        num_warps = 8
    elif block_size > 4095:
        num_warps=16

    # Init our parallel programs(?)
    grid = (rows,)

    # Allocate or output buffer
    sm_out = torch.empty_like(x)

    _softmax_fwd_kernel[grid](
        sm_out, # Output buffer
        sm_out.stride(0), # Stride to move between rows (not necessarily num columns)
        x, # Input
        x.stride(0), # Input stride
        cols, # For masking
        block_size=block_size,
        num_warps=num_warps
    )

    return sm_out


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')

device = torch.cuda.current_device()
properties = driver.active.utils.get_device_properties(device)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


@triton.jit
def _fused_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax_fused(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = _fused_softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
            # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
            # ISA SECTION (3.6.4 for CDNA3)
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
            # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
            # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
            # not required to be equal numbers of both types.
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2

            # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
            # When we divide this number with WARP_SIZE we get maximum number of waves that can
            # execute on a CU (multi-processor)  in parallel.
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y

# def online_softmax(x: torch.Tensor):
#     """ pytorch online softmax """
#     row_count, col_count = x.shape
    
#     output = torch.zeros_like(x)

#     for r in range(row_count):
#         row_max = torch.tensor(0) # m
#         normalizer = torch.tensor(0) # l
#         for c in range(col_count):
#             curr = x[r, c]
#             prev_old_max = row_max
#             row_max = torch.maximum(curr, row_max)
#             # if row_max > prev_old_max:
#             #     print(f"Updated row max is now {row_max}, row = {r}")
            
#             normalizer = normalizer * torch.exp(prev_old_max - row_max) + torch.exp(curr - row_max)
#         output[r,:] = torch.exp(x[r,:] - row_max) / normalizer
#     return output

@triton.jit
def _online_softmax_kernel(
    output_ptr, input_ptr, 
    input_row_stride, output_row_stride, 
    n_rows, n_cols,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    num_stages: tl.constexpr):
    # starting row of the program
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        # The stride represents how much we need to increase the pointer to advance 1 row
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # row in a single block
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_max = 0
        normalizer = 0

        for curr in row:
            prev_row_max = row_max
            if row_max < curr:
                row_max = curr
            normalizer = normalizer * tl.exp(prev_row_max - row_max) + tl.exp(curr - row_max)
        softmax_output = tl.exp(row - row_max) / normalizer
        # Write back output to DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax_online(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (`num_warps`) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 8

    # Number of software piepling stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = _online_softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        if is_hip():
            # NUM_REGS represents the number of regular purpose registers. On CDNA architectures this is half of all registers available.
            # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
            # ISA SECTION (3.6.4 for CDNA3)
            # VGPRs are allocated out of two pools: regular VGPRs and accumulation VGPRs. Accumulation VGPRs are used
            # with matrix VALU instructions, and can also be loaded directly from memory. A wave may have up to 512 total
            # VGPRs, 256 of each type. When a wave has fewer than 512 total VGPRs, the number of each type is flexible - it is
            # not required to be equal numbers of both types.
            if is_cdna():
                NUM_GPRS = NUM_REGS * 2

            # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
            # When we divide this number with WARP_SIZE we get maximum number of waves that can
            # execute on a CU (multi-processor)  in parallel.
            MAX_NUM_THREADS = properties["max_threads_per_sm"]
            max_num_waves = MAX_NUM_THREADS // WARP_SIZE
            occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
        else:
            occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_rows,
        n_cols,
    )
    return y

def benchmark_softmax(fnctn, sample, reference=None):      
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True) 

    start.record()
    result = fnctn(sample)

    end.record()
    torch.cuda.synchronize()
    print(f"{fnctn.__name__} time: {start.elapsed_time(end)}")

    if reference is not None:
        assert torch.allclose(reference, result)
    print(result)

# def main():
#     sample = torch.randn([2, 5], dtype=torch.float32, device="cuda")
    
#     reference = benchmark_softmax(softmax_pt, sample)
    
#     # Naive
#     benchmark_softmax(softmax_naive, sample, reference)

#     # Triton Impl
#     benchmark_softmax(softmax_triton, sample, reference)

#     # Triton Fused Softmax
#     benchmark_softmax(softmax_fused, sample, reference)

#     benchmark_softmax(softmax_online, sample, reference)
# if __name__ == "__main__":
#     main()

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['torch', 'naive', 'triton', 'triton_online'],  # possible values for `line_arg``
        line_names=[
            "Torch",
            "Naive",
            "Triton",
            "Triton Online"
        ],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('yellow', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: softmax_pt(x))
    if provider == 'naive':
        ms = triton.testing.do_bench(lambda: softmax_pt(x))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax_triton(x))
    if provider == 'triton_fused':
        ms = triton.testing.do_bench(lambda: softmax_fused(x))
    if provider == 'triton_online':
        ms = triton.testing.do_bench(lambda: softmax_online(x))
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


if __name__ == "__main__":
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #     benchmark.run(show_plots=True, print_data=True)
    # prof.export_chrome_trace("softmax_trace.json")
    benchmark.run(show_plots=True, print_data=True)