"""
# Following the following tutorial
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#sphx-glr-getting-started-tutorials-03-matrix-multiplication-py

# Blocekd algorithm to multiply (M, K) by a (K, N) matrix
"""

# # Do in parallel
# for m in range(0, M, BLOCK_SIZE_M):
#   # Do in parallel
#   for n in range(0, N, BLOCK_SIZE_N):
#     acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#     for k in range(0, K, BLOCK_SIZE_K):
#       a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#       b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#       acc += dot(a, b)
#     C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc

"""
Difficulty comes from computation of memroy locations for A and B

# Pointer Arithmetic

for row-major 2D tensor X, location of X[i, j] is given by
 &X[i, j] = X + i*stride_xi + j*stride_xj

blocks of pointers for A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] 
and B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N] can be defined 
in pseudo-code as:
"""

# &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
# &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);

"""
# Note we need a modulo to handle case where M is not a multiple of BLOCK_SIZE_M
# or likewise for N. In this case we pad the data. For the K dimension, we will
# handle that later using masking load semantics
"""

# offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
# offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
# offs_k = tl.arange(0, BLOCK_SIZE_K)
# a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
# b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)

"""
Then update inner loop as follow 
"""

# a_ptrs += BLOCK_SIZE_K * stride_ak;
# b_ptrs += BLOCK_SIZE_K * stride_bk;

"""
# L2 Cache Optimization

Each program computes [BLOCK_SIZE_M, BLOCK_SIZE_N] block of C.
Order can affect cache hit rate, and row major won't be optimized
"""

# pid = tl.program_id(axis=0)
# grid_n = tl.cdiv(N, BLOCK_SIZE_N)
# pid_m = pid // grid_n
# pid_n = pid % grid_n

"""
This will need to load the entirety of the B matrix before continuing to
the next row of A. Instead, we can use cached elements of B by 'super-grouping'
blocks in groups of GROUP_M rows before switching to the next column

"""

# # Program ID
# pid = tl.program_id(axis=0)
# # Number of program ids along the M axis
# num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
# # Number of programs ids along the N axis
# num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
# # Number of programs in group
# num_pid_in_group = GROUP_SIZE_M * num_pid_n
# # Id of the group this program is in
# group_id = pid // num_pid_in_group
# # Row-id of the first program in the group
# first_pid_m = group_id * GROUP_SIZE_M
# # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
# group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
# # *Within groups*, programs are ordered in a column-major order
# # Row-id of the program in the *launch grid*
# pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
# # Col-id of the program in the *launch grid*
# pid_n = (pid % num_pid_in_group) // group_size_m


import torch
import triton

import triton.language as tl

def torch_matmul(A, B):
    return A @ B


def naive_matmul(A, B):
    assert A.dim() == 2
    assert B.dim() == 2
    M, K = A.shape
    N = B.shape[1]

    # TODO: Harcoded blocks
    block_M = M // 3
    block_N = N // 3
    block_K = K // 3

    output = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    for start_M in range(0, M, block_M):
        stop_M = start_M + block_M

        for start_N in range(0, N, block_N):
            stop_N = start_N + block_N

            accum = torch.zeros((block_M, block_N), dtype=torch.float32, device="cuda")
            for start_K in range(0, K, block_K):
                stop_K = start_K + block_K

                tileA = A[start_M:stop_M, start_K:stop_K]
                tileB = B[start_K:stop_K, start_N:stop_N]
                accum += tileA @ tileB
            output[start_M:stop_M, start_N:stop_N] = accum
    return output

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=0),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=0),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Poitners to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables reperesent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, # Number of A rows to do in each group
    ACTIVATION: tl.constexpr
):
    """
    Kernel for mamul C = aXB
    A is (M, K), B is (K, N) and C is (M, N)
    """
    # Map program ids `pid` to the block of C it should compute
    # This is done in a grouped ordering ot promote L2 data reuse
    # See above `L2 Cache Optimizations` section for details
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M) # Ceiling div, e.g. (5, 2) --> 3
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n # Num groups (M blocks processed together) * num B cols in block
    group_id = pid // num_pid_in_group 
    first_pid_m = group_id * GROUP_SIZE_M # The first pid in the group
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M) # group size is min of GROUP_SIZE_M or elems in the group
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m) # Get pid_m, second mod to account for out of bound
    pid_n = (pid % num_pid_in_group) // group_size_m # PID N only gows up to number of N Blocks, so mod. Then div in case it was reduced

    # Create pointers for the first blocks of A and B
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M # Relative pointers (no stride, and modulo for oob)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N # Ditto
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak) # Get A pointers
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn) # Get B pointers

    # Iterate to compute a block of the C matrix
    # We accumulate into a `BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy
    # `accumulator` will be converted back to fp16 after the loop

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # Init accumulator
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)): # For each K block
        # Load the next block of A and B, generate a mask by checking the K dimension
        # If OOB set it to 0
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0) # Load the a block
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0) # Load the b block
        # We accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator) # Accumulate the block matmul
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak  
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Arbitrary activation fusion
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # Write back the block to the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)



    
# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


torch.manual_seed(0)
a = torch.randn((512, 512), device='cuda', dtype=torch.float16)
b = torch.randn((512, 512), device='cuda', dtype=torch.float16)
triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output_with_fp16_inputs={triton_output}")
print(f"torch_output_with_fp16_inputs={torch_output}")
# Bigger tolerance for AMD MI200 devices.
# MI200 devices use reduced precision fp16 and bf16 and flush input and
# output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
rtol = 1e-2 if is_hip_mi200() else 0
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")

TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
if TORCH_HAS_FP8 and is_cuda():
    torch.manual_seed(0)
    a = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    b = torch.randn((512, 512), device="cuda", dtype=torch.float16)
    a = a.to(torch.float8_e5m2)
    # pre-transpose b for efficiency.
    b = b.T
    b = b.to(torch.float8_e5m2)
    triton_output = matmul(a, b)
    torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))
    print(f"triton_output_with_fp8_inputs={triton_output}")
    print(f"torch_output_with_fp8_inputs={torch_output}")
    if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")

ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

configs = []
for fp8_inputs in [False, True]:
    if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
        continue
    configs.append(
        triton.testing.Benchmark(
            x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
            x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
            line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
            # Possible values for `line_arg`
            # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
            line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
            line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
            styles=[("green", "-"), ("blue", "-")],
            ylabel="TFLOPS",  # Label name for the y-axis
            plot_name="matmul-performance-" +
            ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
            args={"fp8_inputs": fp8_inputs},
        ))


@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, fp8_inputs):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if TORCH_HAS_FP8 and fp8_inputs:
        a = a.to(torch.float8_e5m2)
        b = b.T
        b = b.to(torch.float8_e5m2)
    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(show_plots=True, print_data=True)

# def benchmark_matmul(fnctn, A, B, reference=None):      
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True) 

#     start.record()
#     result = fnctn(A, B)

#     end.record()
#     torch.cuda.synchronize()
#     print(f"{fnctn.__name__} time: {start.elapsed_time(end)}")

#     if reference is not None:
#         assert torch.allclose(reference, result)

# def main():
#     A = torch.randn([24, 15], dtype=torch.float32, device="cuda")
#     B = torch.randn([15, 30], dtype=torch.float32, device="cuda")
    
#     reference = benchmark_matmul(torch_matmul, A, B)
    
#     # Naive
#     benchmark_matmul(naive_matmul, A, B, reference)

# if __name__ == "__main__":
#     main()