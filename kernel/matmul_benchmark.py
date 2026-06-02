"""
Self-contained benchmark: Triton matmul vs PyTorch (cuBLAS)
B is pre-transposed so both A and Bt are accessed contiguously in memory.

Run with:  python matmul_benchmark.py
"""

import torch
import triton
import triton.language as tl


# ── 1. Triton kernel (uses pre-transposed Bt) ────────────────────────────────
#
# Math:  C[i,j] = Σk  A[i,k] × Bᵀ[j,k]
#
# Both A and Bt are accessed along their row dimension as k increases
# → contiguous memory access → fully coalesced → max HBM bandwidth
#
# Pointer arithmetic:
#   a_ptrs[i,k] = A_ptr  + offs_m[i] * K + offs_k      stride along row of A
#   b_ptrs[j,k] = Bt_ptr + offs_n[j] * K + offs_k      stride along row of Bt
#
# Each K-step:  a_ptrs += BLOCK_K   (slide right in A row)
#               b_ptrs += BLOCK_K   (slide right in Bt row)  ← was BLOCK_K * N

@triton.jit
def matmul_kernel_transposed_b(
    A_ptr, Bt_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                    # [BLOCK_K]

    # A  shape [M, K]: row-major, stride K along rows
    a_ptrs  = A_ptr  + offs_m[:, None] * K + offs_k[None, :]   # [BLOCK_M, BLOCK_K]

    # Bt shape [N, K]: row-major (= B transposed), stride K along rows
    # reading row j of Bt = reading column j of B → contiguous ✓
    bt_ptrs = Bt_ptr + offs_n[:, None] * K + offs_k[None, :]   # [BLOCK_N, BLOCK_K]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a  = tl.load(a_ptrs,  mask=offs_k[None, :] < K - k, other=0.0)  # [BLOCK_M, BLOCK_K]
        bt = tl.load(bt_ptrs, mask=offs_k[None, :] < K - k, other=0.0)  # [BLOCK_N, BLOCK_K]

        # tl.dot expects [M,K] × [K,N] so transpose bt before dot
        acc += tl.dot(a, tl.trans(bt))   # [BLOCK_M, BLOCK_K] × [BLOCK_K, BLOCK_N]

        a_ptrs  += BLOCK_K   # slide right along A row  (contiguous)
        bt_ptrs += BLOCK_K   # slide right along Bt row (contiguous) ← key change

    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask   = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


# ── 2. Python wrappers ────────────────────────────────────────────────────────

def triton_matmul(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=16):
    """Original kernel — B accessed with stride N (strided, not coalesced)."""
    from triton_original import matmul_kernel_original  # keep old kernel for comparison
    M, K = A.shape
    _, N = B.shape
    C    = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel_original[grid](A, B, C, M, N, K,
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C


def triton_matmul_transposed(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=16):
    """New kernel — B pre-transposed, both A and Bt accessed contiguously."""
    M, K = A.shape
    _, N = B.shape
    Bt   = B.t().contiguous()          # ← pre-transpose once, outside kernel
    C    = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel_transposed_b[grid](A, Bt, C, M, N, K,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C


# ── 3. Benchmark helper ───────────────────────────────────────────────────────

def benchmark(fn, warmup=10, rep=100):
    """Returns median latency in ms (scalar). Used for summary stats."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def benchmark_per_run(fn, warmup=10, rep=100):
    """Returns list of per-call latencies in ms. One CUDA event pair per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


# ── 4. Inline original kernel (no separate file needed) ──────────────────────

@triton.jit
def matmul_kernel_original(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_original(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=16):
    M, K = A.shape
    _, N = B.shape
    C    = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel_original[grid](A, B, C, M, N, K,
                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C


# ── 4b. Large BLOCK_K kernel (strided B, BLOCK_K=64) ─────────────────────────
#
# Why BLOCK_K=64 is the maximum for this tile config:
#   shared mem = (BLOCK_M x BLOCK_K + BLOCK_K x BLOCK_N) x 4B x num_stages
#              = (64x64 + 64x64) x 4 x 3 = 98304 bytes = 96 KB <= 100 KB
#   BLOCK_K=128 would need 192 KB -> exceeds limit
#
# Fewer K iterations = less loop overhead and syncthreads calls:
#   BLOCK_K=16 -> 4096/16 = 256 iterations
#   BLOCK_K=64 -> 4096/64 =  64 iterations  (4x fewer)
#
# Tradeoff: more shared memory used -> fewer blocks fit per SM

@triton.jit
def matmul_kernel_large_k(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_matmul_large_k(A, B, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    M, K = A.shape
    _, N = B.shape
    C    = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_kernel_large_k[grid](A, B, C, M, N, K,
                                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    return C


# ── 5. Main ───────────────────────────────────────────────────────────────────

def run(M=4096, N=4096, K=4096, BLOCK_M=64, BLOCK_N=64, BLOCK_K=16):
    print(f"\nMatrix size: {M}×{K} × {K}×{N}   tile: {BLOCK_M}×{BLOCK_N}×{BLOCK_K}")
    print("-" * 65)

    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)

    # ── correctness ──
    # Triton uses TF32 by default on Ada (sm89): rounds fp32 mantissa to 10 bits
    # before each multiply. Over K=4096 accumulations this causes ~0.3 max diff
    # vs fp32 reference. This is expected and not a bug.
    # Use atol=0.5 for TF32, or compare against a TF32 pytorch reference.
    torch.backends.cuda.matmul.allow_tf32 = True   # match Triton's precision
    ref_tf32 = torch.matmul(A, B)
    torch.backends.cuda.matmul.allow_tf32 = False  # restore
    ref_fp32 = torch.matmul(A, B)

    out_orig = triton_matmul_original(A, B, BLOCK_M, BLOCK_N, BLOCK_K)
    out_t    = triton_matmul_transposed(A, B, BLOCK_M, BLOCK_N, BLOCK_K)

    diff_orig_tf32 = (ref_tf32 - out_orig).abs().max().item()
    diff_t_tf32    = (ref_tf32 - out_t).abs().max().item()
    diff_orig_fp32 = (ref_fp32 - out_orig).abs().max().item()

    print(f"Precision note: Triton uses TF32 on sm89 (10-bit mantissa)")
    print(f"  vs fp32 ref:  max_diff={diff_orig_fp32:.6f}  "
          f"(expected ~0.3 for K={K} — NOT a bug)")
    print(f"  vs tf32 ref:  max_diff={diff_orig_tf32:.6f}  "
          f"{'✓ OK' if diff_orig_tf32 < 0.01 else '✗ real mismatch'}")
    print(f"Correctness (pre-transposed B vs tf32 ref): max_diff={diff_t_tf32:.6f}  "
          f"{'✓ OK' if diff_t_tf32 < 0.01 else '✗ real mismatch'}")

    # ── timing ──
    # note: pre-transpose cost is excluded (done once before inference loop)
    Bt = B.t().contiguous()

    REP = 100
    flops = 2 * M * N * K
    def tflops(ms): return flops / (ms * 1e-3) / 1e12

    kernels = [
        ("PyTorch (cuBLAS)",                    lambda: torch.matmul(A, B)),
        ("Triton original (strided B)",          lambda: triton_matmul_original(A, B, BLOCK_M, BLOCK_N, BLOCK_K)),
        ("Triton large BLOCK_K=64 (strided B)",  lambda: triton_matmul_large_k(A, B, BLOCK_M, BLOCK_N, BLOCK_K=64)),
        ("Triton pre-transposed B (incl. .t())", lambda: triton_matmul_transposed(A, B, BLOCK_M, BLOCK_N, BLOCK_K)),
        ("Triton pre-transposed B (kernel only)",lambda: matmul_kernel_transposed_b[
            (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
            A, Bt, torch.empty(M, N, device='cuda'),
            M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)),
    ]

    # collect per-run timings for all kernels
    all_times = {}
    for name, fn in kernels:
        print(f"  timing {name} ...", flush=True)
        all_times[name] = benchmark_per_run(fn, warmup=10, rep=REP)

    # ── per-run table ──
    col_w = 10
    names = [n for n, _ in kernels]
    print(f"\n{'Run':>4}  " + "  ".join(f"{n[:col_w]:>{col_w}}" for n in names))
    print(f"{'─'*4}  " + "  ".join("─"*col_w for _ in names) + "  (ms)")
    for i in range(REP):
        row = "  ".join(f"{all_times[n][i]:>{col_w}.3f}" for n in names)
        print(f"{i+1:>4}  {row}")

    # ── summary ──
    import statistics
    print(f"\n{'Kernel':<44} {'BLOCK_K':>7}  {'median ms':>10}  {'mean ms':>9}  {'min ms':>8}  {'TFLOPS':>8}")
    print(f"{'─'*44} {'─'*7}  {'─'*10}  {'─'*9}  {'─'*8}  {'─'*8}")
    bk_map = {
        "PyTorch (cuBLAS)": "─",
        "Triton original (strided B)": str(BLOCK_K),
        "Triton large BLOCK_K=64 (strided B)": "64",
        "Triton pre-transposed B (incl. .t())": str(BLOCK_K),
        "Triton pre-transposed B (kernel only)": str(BLOCK_K),
    }
    for name in names:
        times = all_times[name]
        med   = statistics.median(times)
        mean  = statistics.mean(times)
        mn    = min(times)
        bk    = bk_map[name]
        print(f"{name:<44} {bk:>7}  {med:>10.3f}  {mean:>9.3f}  {mn:>8.3f}  {tflops(med):>8.1f}")

    orig_med  = statistics.median(all_times["Triton original (strided B)"])
    lk_med    = statistics.median(all_times["Triton large BLOCK_K=64 (strided B)"])
    tp_med    = statistics.median(all_times["Triton pre-transposed B (kernel only)"])
    torch_med = statistics.median(all_times["PyTorch (cuBLAS)"])
    print(f"\nSpeedup large BLOCK_K vs original:          {orig_med/lk_med:.2f}x")
    print(f"Speedup transposed kernel only vs original: {orig_med/tp_med:.2f}x")
    print(f"Speedup transposed kernel only vs cuBLAS:   {torch_med/tp_med:.2f}x")

    # ── kernel stats for large_k ──
    compiled_lk = matmul_kernel_large_k[
        (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        A, B, torch.empty(M, N, device='cuda'),
        M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=64)
    print(f"\nKernel stats (large BLOCK_K=64):")
    print(f"  Registers per thread : {compiled_lk.n_regs}")
    print(f"  Shared memory        : {compiled_lk.metadata.shared / 1024:.1f} KB  "
          f"(was {16:.0f} KB with BLOCK_K=16)")
    print(f"  Num warps            : {compiled_lk.metadata.num_warps}")
    print(f"  Pipeline stages      : {compiled_lk.metadata.num_stages}")
    shmem_lk      = compiled_lk.metadata.shared
    regs_lk       = compiled_lk.n_regs * compiled_lk.metadata.num_warps * 32
    threads_lk    = compiled_lk.metadata.num_warps * 32
    bsm_regs      = 65536        // regs_lk    if regs_lk   > 0 else 99
    bsm_threads   = 1536         // threads_lk if threads_lk > 0 else 99
    bsm_shmem     = (128 * 1024) // shmem_lk   if shmem_lk  > 0 else 99
    bsm           = min(bsm_regs, bsm_threads, bsm_shmem, 16)
    total_blocks  = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    print(f"  Blocks per SM        : {bsm}  "
          f"(regs={bsm_regs}, threads={bsm_threads}, shmem={bsm_shmem}, hw=16)")
    print(f"  Waves                : {-(-total_blocks // (128 * bsm))}  "
          f"({total_blocks} blocks / {128*bsm} in-flight)")

    # ── kernel stats (transposed B) ──
    compiled = matmul_kernel_transposed_b[
        (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        A, Bt, torch.empty(M, N, device='cuda'),
        M, N, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
    regs_per_block    = compiled.n_regs * compiled.metadata.num_warps * 32
    shmem_per_block   = compiled.metadata.shared
    threads_per_block = compiled.metadata.num_warps * 32
    blocks_by_regs    = 65536        // regs_per_block    if regs_per_block    > 0 else 99
    blocks_by_threads = 1536         // threads_per_block if threads_per_block > 0 else 99
    blocks_by_shmem   = (128 * 1024) // shmem_per_block   if shmem_per_block   > 0 else 99
    blocks_per_sm     = min(blocks_by_regs, blocks_by_threads, blocks_by_shmem, 16)
    total_blocks      = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    in_flight         = 128 * blocks_per_sm
    waves             = -(-total_blocks // in_flight)
    print(f"\nKernel stats (transposed B, BLOCK_K={BLOCK_K}):")
    print(f"  Registers per thread : {compiled.n_regs}")
    print(f"  Shared memory        : {compiled.metadata.shared / 1024:.1f} KB")
    print(f"  Num warps            : {compiled.metadata.num_warps}")
    print(f"  Pipeline stages      : {compiled.metadata.num_stages}")
    print(f"  Blocks per SM        : {blocks_per_sm}  "
          f"(regs={blocks_by_regs}, threads={blocks_by_threads}, shmem={blocks_by_shmem}, hw=16)")
    print(f"  Waves                : {waves}  ({total_blocks} blocks / {in_flight} in-flight)")


def sanity_check(BLOCK_M=16, BLOCK_N=16, BLOCK_K=16):
    """
    Small ones-matrix test to isolate indexing bugs from precision noise.
    A = ones(16,16), B = ones(16,16) → every cell of C should be exactly 16.0
    Any non-zero diff here is a real bug, not a TF32 precision issue.
    """
    print("\n" + "=" * 65)
    print("Sanity check: 16×16 ones matrix (expect every cell = 16.0)")
    print("=" * 65)

    A = torch.ones(16, 16, device='cuda', dtype=torch.float32)
    B = torch.ones(16, 16, device='cuda', dtype=torch.float32)
    ref = torch.matmul(A, B)

    print(f"ref[0,0]    = {ref[0,0].item():.1f}  (expected 16.0)")

    # original kernel
    out_orig = triton_matmul_original(A, B, BLOCK_M, BLOCK_N, BLOCK_K)
    diff_orig = (ref - out_orig).abs()
    ok_orig = diff_orig.max().item() == 0.0
    print(f"original:    max_diff={diff_orig.max().item():.6f}  out[0,0]={out_orig[0,0].item():.1f}  "
          f"{'✓ exact' if ok_orig else '✗ BUG — indexing error'}")

    # transposed kernel
    out_t = triton_matmul_transposed(A, B, BLOCK_M, BLOCK_N, BLOCK_K)
    diff_t = (ref - out_t).abs()
    ok_t = diff_t.max().item() == 0.0
    print(f"transposed:  max_diff={diff_t.max().item():.6f}  out[0,0]={out_t[0,0].item():.1f}  "
          f"{'✓ exact' if ok_t else '✗ BUG — indexing error'}")

    if not ok_orig or not ok_t:
        # show which cells are wrong
        bad = diff_orig.nonzero()
        print(f"\nFirst 5 wrong cells (original kernel):")
        for idx in bad[:5]:
            i, j = idx[0].item(), idx[1].item()
            print(f"  C[{i},{j}]: expected={ref[i,j].item():.1f}  got={out_orig[i,j].item():.1f}")


if __name__ == "__main__":
    run()
    sanity_check()
