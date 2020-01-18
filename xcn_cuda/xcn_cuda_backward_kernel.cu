#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <ATen/AccumulateType.h>            // acc_type, accscalar_t
#include <THC/THCDeviceUtils.cuh>           // WARP_SHFL_XOR
#include <DeviceSqrt.cuh>                   // device_sqrt
using namespace torch;

#if defined(__HIP_PLATFORM_HCC__)
constexpr int WARP_SIZE = 64;
#else
constexpr int WARP_SIZE = 32;
#endif

// The maximum number of threads in a block
#if defined(__HIP_PLATFORM_HCC__)
constexpr int MAX_BLOCK_SIZE = 256;
#else
constexpr int MAX_BLOCK_SIZE = 512;
#endif

// Number of threads in a block given an input size up to MAX_BLOCK_SIZE
static int getNumThreads(int nElem) {
#if defined(__HIP_PLATFORM_HCC__)
  int threadSizes[5] = { 16, 32, 64, 128, MAX_BLOCK_SIZE };
#else
  int threadSizes[5] = { 32, 64, 128, 256, MAX_BLOCK_SIZE };
#endif
  for (int i = 0; i != 5; ++i) {
    if (nElem <= threadSizes[i]) {
      return threadSizes[i];
    }
  }
  return MAX_BLOCK_SIZE;
}

// Returns the index of the most significant 1 bit in `val`.
__device__ __forceinline__ int getMSB(int val) {
  return 31 - __clz(val);
}

template <typename scalar_t, typename accscalar_t>
struct Float2 {
  accscalar_t v1, v2;
  __device__ Float2() {}
  __device__ Float2(scalar_t v1, scalar_t v2) : v1(static_cast<accscalar_t>(v1)), v2(static_cast<accscalar_t>(v2)) {}
  __device__ Float2(int v) : v1(static_cast<accscalar_t>(v)), v2(static_cast<accscalar_t>(v)) {}
  __device__ Float2& operator+=(const Float2& a) {
    v1 += a.v1;
    v2 += a.v2;
    return *this;
  }
};

template <typename scalar_t, typename accscalar_t, typename PTA>
struct GradOp {
  __device__ GradOp(accscalar_t m, const PTA& i, const PTA& g)
    : mean(m), input(i), grad_output(g) {}
  __device__ __forceinline__ Float2<scalar_t, accscalar_t> operator()(int batch, int plane, int n) {
    accscalar_t g = grad_output[batch][plane][n];
    accscalar_t c = static_cast<accscalar_t>(input[batch][plane][n]) - mean;
    return Float2<scalar_t, accscalar_t>(g, g * c);
  }
  const accscalar_t mean;
  const PTA& input;
  const PTA& grad_output;
};

// Sum across all threads within a warp
template <typename T>
static __device__ __forceinline__ T warpSum(T val) {
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    val += WARP_SHFL_XOR(val, 1 << i, WARP_SIZE);
  }
  return val;
}

template <typename scalar_t, typename accscalar_t>
static __device__ __forceinline__ Float2<scalar_t, accscalar_t> warpSum(Float2<scalar_t, accscalar_t> value) {
  value.v1 = warpSum(value.v1);
  value.v2 = warpSum(value.v2);
  return value;
}

// Sum across (batch, x/y/z) applying Op() pointwise
// this works by first having each thread sum it's part
// of the data. Then there is a double-shuffeling reduction.
// First each warp (of WARP_SIZE threads) uses warpSum to reduce its
// data to the "warp leader", who writes its value into shared memory.
// Then a single warp reads the remaining (at most WARP_SIZE) items
// and reduces them using another warpSum.
// The implicit assumption is that there are no more
// than WARP_SIZE**2 threads.
template<typename scalar_t, typename Op, typename PTA>
__device__ scalar_t reduce(Op op, PTA tensor, int plane) {
  // first the reductions each thread does separately
  scalar_t sum = static_cast<scalar_t>(0);
  for (int batch = threadIdx.y; batch < tensor.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < tensor.size(2); x += blockDim.x) {
      sum += op(batch, plane, x);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  sum = warpSum(sum);

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __shared__ scalar_t shared[WARP_SIZE];
  __syncthreads();
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  if (tid % WARP_SIZE == 0) {
    shared[tid / WARP_SIZE] = sum;
  }
  if (tid >= blockDim.x * blockDim.y / WARP_SIZE && tid < WARP_SIZE) {
    // zero out the other entries in shared
    shared[tid] = (scalar_t)0;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid / WARP_SIZE == 0) {
    sum = warpSum(shared[tid]);
    if (tid == 0) {
      shared[0] = sum;
    }
  }
  __syncthreads();

  // Everyone picks it up, should be broadcast into the whole grad_input
  return shared[0];
}

template<typename T>
struct Control {
  __device__ __forceinline__ bool operator()(T invstd, double minl) const {
    return (invstd * minl < static_cast<T>(1));
  }
};

// DefaultPtrTraits
template <typename scalar_t>
__global__ void batch_norm_backward_kernel(
    const PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, size_t> input,
    PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, size_t> grad_output,
    PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, size_t> grad_input,
    PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, size_t> grad_weight,
    PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, size_t> grad_bias,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, size_t> weight,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, size_t> save_mean,
    const PackedTensorAccessor<scalar_t, 1, DefaultPtrTraits, size_t> save_invstd,
    double minl) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  int plane = blockIdx.x;
  int N = grad_output.size(0) * grad_output.size(2);

  accscalar_t mean, invstd;
//   if (train) {
    mean = static_cast<accscalar_t>(save_mean[plane]);
    invstd = static_cast<accscalar_t>(save_invstd[plane]);
//   } else {
//     mean = static_cast<accscalar_t>(running_mean[plane]);
//     invstd = static_cast<accscalar_t>(1) / device_sqrt(static_cast<accscalar_t>(running_var[plane]) + epsilon);
//   }

  accscalar_t weight_val = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : accscalar_t(1);
  accscalar_t norm = accscalar_t(1) / N;

  // Compute two values across (batch, x/y/z) in one pass:
  // 1. Sum(grad_output)
  // 2. DotProduct(input - mean, grad_output)
  GradOp<scalar_t, accscalar_t, PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, size_t>> g(mean, input, grad_output);
  Float2<scalar_t, accscalar_t> res = reduce<Float2<scalar_t, accscalar_t>, GradOp<scalar_t, accscalar_t,
                                                                                   PackedTensorAccessor<scalar_t, 3, DefaultPtrTraits, size_t>>>(g, grad_output, plane);
  accscalar_t grad_output_sum = res.v1;
  accscalar_t dot_p = res.v2;

  accscalar_t grad_mean = grad_output_sum * norm;
  accscalar_t proj_scale = dot_p * norm * invstd * invstd;
  accscalar_t grad_scale = invstd * weight_val;

  if (grad_input.data() != NULL) {
    for (int batch = threadIdx.y; batch < grad_output.size(0); batch += blockDim.y) {
      for (int x = threadIdx.x; x < grad_output.size(2); x += blockDim.x) {
        scalar_t go = grad_output[batch][plane][x];
        if(Control<accscalar_t>{}(invstd, minl)){
        // if (train) {
          scalar_t inp = input[batch][plane][x];
          accscalar_t proj = (inp - mean) * proj_scale;
          grad_input[batch][plane][x] = static_cast<scalar_t>((go - proj - grad_mean) * grad_scale);
        // } else {
        //   grad_input[batch][plane][x] = static_cast<scalar_t>(go * grad_scale);
        // }
        } else{
          grad_input[batch][plane][x] = static_cast<scalar_t>((go - grad_mean) * grad_scale);
        }
      }
    }
  }

  if (grad_weight.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_weight[plane] = static_cast<scalar_t>(dot_p * invstd);
    }
  }

  if (grad_bias.size(0) > 0) {
    if (threadIdx.x == 0) {
      grad_bias[plane] = static_cast<scalar_t>(grad_output_sum);
    }
  }
}


std::vector<Tensor> bcn_cuda_backward(
    Tensor grad_out_, const Tensor input_, const Tensor weight_,
    const Tensor save_mean_, const Tensor save_invstd_,
    double minl, std::array<bool,3> grad_input_mask) {

  Tensor grad_input_;
  Tensor grad_input_reshaped;
  Tensor grad_weight_;
  Tensor grad_bias_;
  auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1});
  auto grad_output_reshaped = grad_out_.reshape(input_reshaped.sizes());

  if (grad_input_mask[0]) {
    grad_input_ = at::empty_like(input_);
    grad_input_reshaped = grad_input_.view(input_reshaped.sizes());
  }
  if (grad_input_mask[1]) {
    grad_weight_ = at::empty_like(weight_);
  }
  if (grad_input_mask[2]) {
    grad_bias_ = at::empty_like(weight_);
  }

  dim3 blocks(input_reshaped.size(1));
  int tf = getNumThreads(input_reshaped.size(2));
  dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));
  
  AT_DISPATCH_FLOATING_TYPES(input_.scalar_type(), "batch_norm_backward_cuda", [&] {
    batch_norm_backward_kernel<scalar_t> <<<blocks, threads>>>(
        input_reshaped.packed_accessor<scalar_t, 3, DefaultPtrTraits, size_t>(), 
        grad_output_reshaped.packed_accessor<scalar_t, 3, DefaultPtrTraits, size_t>(), 
        grad_input_reshaped.packed_accessor<scalar_t, 3, DefaultPtrTraits, size_t>(), 
        grad_weight_.packed_accessor<scalar_t, 1, DefaultPtrTraits, size_t>(), 
        grad_bias_.packed_accessor<scalar_t, 1, DefaultPtrTraits, size_t>(), 
        weight_.packed_accessor<scalar_t, 1, DefaultPtrTraits, size_t>(),
        save_mean_.packed_accessor<scalar_t, 1, DefaultPtrTraits, size_t>(), 
        save_invstd_.packed_accessor<scalar_t, 1, DefaultPtrTraits, size_t>(),
        minl);
  });
//   THCudaCheck(cudaGetLastError());

  return {grad_input_, grad_weight_, grad_bias_};
}