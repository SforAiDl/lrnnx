import triton
import triton.language as tl
from packaging import version

# Standardize version check to handle Sphinx mocking
try:
    # If triton is mocked by Sphinx, __version__ will be a Mock object, not a string
    triton_version = triton.__version__
    if not isinstance(triton_version, str):
        triton_version = "2.1.0" # Default fallback for documentation builds
    TRITON3 = version.parse(triton_version) >= version.parse("3.0.0")
except (TypeError, AttributeError):
    TRITON3 = False

if TRITON3:
    @triton.jit
    def softplus(dt):
        """
        Triton JIT implementation of the softplus activation function.
        
        This branch handles Triton versions >= 3.0.0.

        Args:
            dt: Input tensor value.

        Returns:
            The softplus applied to the input: log(exp(dt) + 1).
        """
        return tl.math.log(tl.math.exp(dt) + 1)
    
else:
    
    @triton.jit
    def softplus(dt):
        """
        Triton JIT implementation of the softplus activation function.

        This branch handles Triton versions < 3.0.0.

        Args:
            dt: Input tensor value.

        Returns:
            The softplus applied to the input using log1p.
        """
        return tl.math.log1p(tl.exp(dt))