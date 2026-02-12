import os

from setuptools import setup
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

simplified_scan_path = os.path.join("csrc", "simplified_scan")
selective_scan_path = os.path.join("csrc", "selective_scan")
s4_kernels_path = os.path.join("csrc", "s4")

ext_modules = []

if CUDA_HOME is not None:
    print(f"CUDA_HOME found at {CUDA_HOME}. Building CUDA extensions.")
    ext_modules = [
        CUDAExtension(
            name="selective_scan_cuda",
            sources=[
                os.path.join(selective_scan_path, "bindings.cpp"),
                os.path.join(selective_scan_path, "selective_scan_cpu.cpp"),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_real_bilinear.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_real_zoh.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_real_mamba.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_real_dirac.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_real_s7.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_real_rglru.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_complex_mamba.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_complex_zoh.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_complex_bilinear.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "forward_kernels/selective_scan_fp32_complex_dirac.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_real_bilinear.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_real_zoh.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_real_mamba.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_real_dirac.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_real_s7.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_real_rglru.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_complex_mamba.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_complex_zoh.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_complex_bilinear.cu",
                ),
                os.path.join(
                    selective_scan_path,
                    "backward_kernels/selective_scan_fp32_complex_dirac.cu",
                ),
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            name="simplified_scan_cuda",
            sources=[
                os.path.join(simplified_scan_path, "bindings.cpp"),
                os.path.join(simplified_scan_path, "simplified_scan_cpu.cpp"),
                os.path.join(
                    simplified_scan_path,
                    "forward_kernels/simplified_scan_fp32_bilinear.cu",
                ),
                os.path.join(
                    simplified_scan_path,
                    "forward_kernels/simplified_scan_fp32_zoh.cu",
                ),
                os.path.join(
                    simplified_scan_path,
                    "forward_kernels/simplified_scan_fp32_dirac.cu",
                ),
                os.path.join(
                    simplified_scan_path,
                    "backward_kernels/simplified_scan_bilinear.cu",
                ),
                os.path.join(
                    simplified_scan_path,
                    "backward_kernels/simplified_scan_zoh.cu",
                ),
                os.path.join(
                    simplified_scan_path,
                    "backward_kernels/simplified_scan_dirac.cu",
                ),
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        ),
        CUDAExtension(
            name="structured_kernels",
            sources=[
                os.path.join(s4_kernels_path, "cauchy.cpp"),
                os.path.join(s4_kernels_path, "cauchy_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-g", "-march=native", "-funroll-loops"],
                "nvcc": ["-O2", "-lineinfo", "--use_fast_math"],
            },
        ),
    ]
else:
    print(
        "CUDA_HOME not found. Skipping CUDA extension build."
    )

setup(
    # loads from pyproject.toml
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
