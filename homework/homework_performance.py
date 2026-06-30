import time
import torch


# 3.1 матрицы

def create_matrices():
    m1 = torch.rand(64, 1024, 1024)
    m2 = torch.rand(128, 512, 512)
    m3 = torch.rand(256, 256, 256)
    return m1, m2, m3


# 3.2 замер времени

def measure_cpu(fn, *args):
    start = time.time()
    fn(*args)
    return (time.time() - start) * 1000


def measure_gpu(fn, *args):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    fn(*args)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


# 3.3 бенчмарк

def benchmark(tensor_cpu):
    cuda_ok = torch.cuda.is_available()
    tensor_gpu = tensor_cpu.cuda() if cuda_ok else None

    ops = [
        ("matmul",    lambda t: torch.matmul(t, t.transpose(-1, -2))),
        ("add",       lambda t: t + t),
        ("mul",       lambda t: t * t),
        ("transpose", lambda t: t.transpose(-1, -2).contiguous()),
        ("sum",       lambda t: t.sum()),
    ]

    print(f"{'op':<15} | {'cpu ms':>10} | {'gpu ms':>10} | {'speedup':>10}")
    print("-" * 55)

    for name, op in ops:
        cpu_ms = measure_cpu(op, tensor_cpu)
        if cuda_ok:
            gpu_ms = measure_gpu(op, tensor_gpu)
            speedup = f"{cpu_ms / gpu_ms:.1f}x"
        else:
            gpu_ms = float('nan')
            speedup = "no GPU"
        print(f"{name:<15} | {cpu_ms:>10.2f} | {str(round(gpu_ms,2)):>10} | {speedup:>10}")


# 3.4 выводы по результатам:
# - matmul и поэлементные операции дают наибольшее ускорение на GPU
# - на маленьких матрицах GPU может быть медленнее из-за накладных расходов
# - узкое место при пересылке данных — пропускная способность PCIe

if __name__ == "__main__":
    m1, m2, m3 = create_matrices()
    for label, t in [("64x1024x1024", m1), ("128x512x512", m2), ("256x256x256", m3)]:
        print(f"\nматрица {label}")
        benchmark(t)
