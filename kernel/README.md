# Custom Triton GEMM vs cuBLAS on RTX 4090

This is a very simple matrix multiplication case:

* A of size 4096*4096
* B of size 4096*4096
* `C = A @ B`

Say we have already rotated B, so that `B^T` has the row vector accessed contiguously in memory (torch does it under the table).

`torch.matmul` uses `cuBLAS` which compute by `64*64` tiles. For `i := n:n+BLOCK_N`, `j := m:m+BLOCK_M`, `l := k:k+BLOCK_K`,

```C[i,j] += Σl  A[i,k] × Bᵀ[j,k]```

GPU can process the blocks in parallel. Each GPU SM processes at most X blocks, limited by the hardware (number of registers, warps available etc.) 

We can implement a Triton kernel that does similar things but with adjusted `BLOCK_K` numbers.

Results show improves latency and FLOPS.

```
Kernel                                       BLOCK_K   median ms    mean ms    min ms    TFLOPS
──────────────────────────────────────────── ───────  ──────────  ─────────  ────────  ────────
PyTorch (cuBLAS)                                   ─       2.404      2.398     2.280      57.2
Triton original (strided B)                       16       1.843      1.865     1.811      74.6
Triton large BLOCK_K=64 (strided B)               64       1.917      1.933     1.912      71.7
Triton pre-transposed B (incl. .t())              16       2.086      2.115     2.071      65.9
Triton pre-transposed B (kernel only)             16       1.867      1.888     1.840      73.6

Speedup large BLOCK_K vs original:          0.96x
Speedup transposed kernel only vs original: 0.99x
Speedup transposed kernel only vs cuBLAS:   1.29x

Kernel stats (large BLOCK_K=64):
  Registers per thread : 155
  Shared memory        : 64.0 KB  (was 16 KB with BLOCK_K=16)
  Num warps            : 4
  Pipeline stages      : 3
  Blocks per SM        : 2  (regs=3, threads=12, shmem=2, hw=16)
  Waves                : 16  (4096 blocks / 256 in-flight)

Kernel stats (transposed B, BLOCK_K=16):
  Registers per thread : 79
  Shared memory        : 16.0 KB
  Num warps            : 4
  Pipeline stages      : 3
  Blocks per SM        : 6  (regs=6, threads=12, shmem=8, hw=16)
  Waves                : 6  (4096 blocks / 768 in-flight)
```
