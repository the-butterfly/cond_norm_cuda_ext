/*
Follow pytorch official tutorial and backend codes.
https://github.com/pytorch/pytorch/blob/877c96cddfebee00385307f9e1b1f3b4ec72bfdc/aten/src/ATen/native/cuda/Normalization.cuh
https://github.com/pytorch/pytorch/blob/877c96cddfebee00385307f9e1b1f3b4ec72bfdc/aten/src/ATen/native/cuda/Normalization.cu
*/

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <ATen/AccumulateType.h>            // acc_type, accscalar_t
#include <THC/THCDeviceUtils.cuh>           // WARP_SHFL_XOR
#include <DeviceSqrt.cuh>               // device_sqrt
using namespace torch;

// aten/src/ATen/native/Normalization.cpp
void check_dims_match_num_input_features(const char* arg_name, int64_t expected, int64_t actual){
  AT_CHECK(actual == expected,
            arg_name, " should contain ", expected, " elements not ", actual);
}

static inline Tensor repeat_if_defined(const Tensor& t, int64_t repeat) {
  if (t.defined()) {
    return t.repeat(repeat);
  }
  return t;
}

// aten/src/ATen/native/cuda/Normalization.cuh
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

template<typename T>
struct InvStd {
  __device__ __forceinline__ T operator()(T var, double minl, double epsilon) const {
    T invstd = 0;
    if (var != static_cast<T>(0) || epsilon != static_cast<T>(0)) {
      invstd = static_cast<T>(1) / fmaxf(device_sqrt(var + epsilon), minl);
    }
    return invstd;
  }
};


template <template<typename T> class VarTransform, typename scalar_t>
__global__ void batch_norm_collect_statistics_kernel(
    const PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, size_t> input,
    PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t> save_mean,
    PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t> save_transformed_var,
    const double minl,
    const double epsilon) {

  __shared__ int shared_n[2 * 2 * WARP_SIZE + WARP_SIZE];

  int plane = blockIdx.x;
  int N = input.size(0) * input.size(2);
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  using stat_accscalar_t = at::acc_type<scalar_t, true>;
  
//   accscalar_t epsilon = static_cast<accscalar_t> eps;

  // Compute the mean and variance across (batch, x/y/z)
  // this uses the Welford (in the for loop)/parallel algorithm (to sum across the block)
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_Online_algorithm
  // and the parallel algorithm on the same page.
  // We use two shuffles to reduce across the entire block.
  // https://devblogs.nvidia.com/faster-parallel-reductions-kepler/ has a description.
  stat_accscalar_t* shared_avg_var = (stat_accscalar_t*) &shared_n[WARP_SIZE];

  // first the reductions each thread does separately
  stat_accscalar_t avg = 0;
  stat_accscalar_t var_n = 0;
  int n = 0;
  for (int batch = threadIdx.y; batch < input.size(0); batch += blockDim.y) {
    for (int x = threadIdx.x; x < input.size(2); x += blockDim.x) {
      stat_accscalar_t v = input[batch][plane][x];
      stat_accscalar_t d1 = v - avg;
      n++;
      avg += d1 / n;
      var_n += d1 * (v - avg);
    }
  }

  // first warpSum to get one value per thread to
  // one value per warp
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // this writes each warps  item into shared memory
  // there are at most WARP_SIZE items left because
  // there are at most WARP_SIZE**2 threads at the beginning
  __syncthreads();
  if (tid % WARP_SIZE == 0) {
    shared_n[tid / WARP_SIZE] = n;
    shared_avg_var[tid / WARP_SIZE * 2] = avg;
    shared_avg_var[tid / WARP_SIZE * 2 + 1] = var_n;
  }
  __syncthreads();
  // now have a second warpSum to reduce the intermediate values
  // from shared memory to a single number. The very first
  // thread writes it to shared memory.

  if (tid < WARP_SIZE) {
    n = (tid < blockDim.x * blockDim.y / WARP_SIZE ? shared_n[tid] : 0);
    avg = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid] : stat_accscalar_t(0));
    var_n = (tid < blockDim.x * blockDim.y  / WARP_SIZE ? shared_avg_var[2 * tid + 1] : stat_accscalar_t(0));
  }
  for (int i = 0; i < getMSB(WARP_SIZE); ++i) {
    stat_accscalar_t o_avg = WARP_SHFL_XOR(avg, 1 << i, WARP_SIZE);
    int o_n = WARP_SHFL_XOR(n, 1 << i, WARP_SIZE);
    stat_accscalar_t factor = 1.0 / fmaxf(1.0, n+o_n);
    var_n += WARP_SHFL_XOR(var_n, 1 << i, WARP_SIZE) + (avg - o_avg) * (avg - o_avg) * n * o_n * factor;
    avg = (n * avg + o_n * o_avg) * factor;
    n += o_n;
  }

  // Save the mean, variance, and moving averages
  if (tid == 0) {
    if (save_mean.data() != NULL) {
      save_mean[plane] = avg;
    }
    if (save_transformed_var.data() != NULL) {
      save_transformed_var[plane] = VarTransform<stat_accscalar_t>{}(var_n / N, minl, epsilon);
    }
  }

}

template <typename scalar_t>
__global__ void batch_norm_transform_input_kernel(
    const PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, size_t> input,
    PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, size_t> output,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t> mean_,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t> var_or_invstd,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t> weight,
    const PackedTensorAccessor<scalar_t, 1, RestrictPtrTraits, size_t> bias,
    double epsilon) {

  using accscalar_t = at::acc_type<scalar_t, true>;
  int plane = blockIdx.x;

  if (plane >= input.size(1)) {
    return;
  }

  accscalar_t gamma = weight.size(0) > 0 ? static_cast<accscalar_t>(weight[plane]) : static_cast<accscalar_t>(1);
  accscalar_t beta = bias.size(0) > 0 ? static_cast<accscalar_t>(bias[plane]) : static_cast<accscalar_t>(0);
  accscalar_t mean = static_cast<accscalar_t>(mean_[plane]);
  accscalar_t invstd;
  invstd = var_or_invstd[plane];

  int bs = input.size(0);
  int fs = input.size(2);

  int bstep  = blockDim.y * gridDim.y;
  for (int batch = threadIdx.y + blockIdx.y * blockDim.y; batch < bs; batch += bstep) {
    auto o = output[batch][plane];
    auto i = input[batch][plane];
    for (int feature = threadIdx.x; feature < fs; feature += blockDim.x) {
      o[feature] = static_cast<scalar_t>(gamma * (i[feature] - mean) * invstd + beta);
    }
  }
}


std::vector<Tensor> bcn_cuda_forward(Tensor input_, Tensor weight_, Tensor bias_, double minl, double epsilon) {

    int64_t n_input = input_.size(1);
    Tensor save_mean_;
    Tensor save_invstd_;
    auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions
    auto output_reshaped = at::empty_like(input_reshaped);

    auto bs = input_reshaped.size(0);
    auto planes = input_reshaped.size(1);
    auto features = input_reshaped.size(2);
    
    auto input_options = input_.options();
    auto input_type = input_reshaped.type();
    if (input_.scalar_type() == at::ScalarType::Half) {
        input_options = input_options.dtype(ScalarType::Float);
    }
    save_mean_ = at::empty({n_input}, input_options);
    save_invstd_ = at::empty({n_input}, input_options);
    
    // The input_transform kernel is pointwise, but we need to balance reading parameters (save_var/mean,
    // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
    // and good occupancy. Quite likely, we could go with even more blocks than 1024.
    // The various planes are independent, so we use blocks for them.
    int tf = std::max<int>(getNumThreads(features/4),
                            std::min<int>(getNumThreads(features), 64));
    int tb = std::max<int>(64/tf, 1);
    dim3 blocks_trans(planes, std::max<int>(1, std::min<int>((256*1024)/planes,
                                                                    (bs+tb-1)/tb)));
    blocks_trans.y = std::min<int>(blocks_trans.y, 65535);
    dim3 threads_trans(tf, tb);
    
    // for the reduction, we cannot use blocks for the batch dim, but if we have few threads in
    // the feature dimension, we'll use some threads for blocks
    dim3 blocks(planes);
    tf = getNumThreads(features);
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

    AT_DISPATCH_FLOATING_TYPES(input_type, "bcn_cuda", [&] {
        batch_norm_collect_statistics_kernel<InvStd, scalar_t> <<<blocks, threads>>>(
            input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, size_t>(), 
            save_mean_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            save_invstd_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(),
            minl, epsilon);
    });

    AT_DISPATCH_FLOATING_TYPES(input_type, "bcn_cuda", [&] {
        batch_norm_transform_input_kernel<scalar_t><<<blocks_trans, threads_trans>>>(
            input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, size_t>(), 
            output_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, size_t>(), 
            save_mean_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            save_invstd_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            weight_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            bias_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            epsilon);
    });
    return {output_reshaped.view(input_.sizes()), save_mean_, save_invstd_};
}

std::vector<Tensor> icn_cuda_forward(Tensor input, Tensor weight, Tensor bias, double minl, double epsilon) {
    int64_t b = input.size(0);
    int64_t c = input.size(1);
    std::vector<int64_t>  shape = {1, b*c, -1};

    Tensor weight_ = repeat_if_defined(weight, b);
    Tensor bias_ = repeat_if_defined(bias, b);

    auto input_reshaped = input.reshape(shape);
    auto output_reshaped = at::empty_like(input_reshaped);

    Tensor save_mean_;
    Tensor save_invstd_;
    // auto input_reshaped = input_.reshape({input_.size(0), input_.size(1), -1}); // internally we merge the feature dimensions

    auto bs = input_reshaped.size(0);
    auto planes = input_reshaped.size(1);
    auto features = input_reshaped.size(2);
    
    auto input_options = input.options();
    auto input_type = input_reshaped.type();
    if (input.scalar_type() == at::ScalarType::Half) {
        input_options = input_options.dtype(ScalarType::Float);
    }
    save_mean_ = at::empty({b*c}, input_options);
    save_invstd_ = at::empty({b*c}, input_options);
    
    // The input_transform kernel is pointwise, but we need to balance reading parameters (save_var/mean,
    // weight/bias) - which we only do once and have a for loop afterwards - with having many threads and blocks
    // and good occupancy. Quite likely, we could go with even more blocks than 1024.
    // The various planes are independent, so we use blocks for them.
    int tf = std::max<int>(getNumThreads(features/4),
                            std::min<int>(getNumThreads(features), 64));
    int tb = std::max<int>(64/tf, 1);
    dim3 blocks_trans(planes, std::max<int>(1, std::min<int>((256*1024)/planes,
                                                                    (bs+tb-1)/tb)));
    blocks_trans.y = std::min<int>(blocks_trans.y, 65535);
    dim3 threads_trans(tf, tb);
    
    // for the reduction, we cannot use blocks for the batch dim, but if we have few threads in
    // the feature dimension, we'll use some threads for blocks
    dim3 blocks(planes);
    tf = getNumThreads(features);
    dim3 threads(tf, std::max<int>(1, MAX_BLOCK_SIZE/tf));

    AT_DISPATCH_FLOATING_TYPES(input_type, "icn_cuda", [&] {
        batch_norm_collect_statistics_kernel<InvStd, scalar_t> <<<blocks, threads>>>(
            input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, size_t>(), 
            save_mean_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            save_invstd_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(),
            minl, epsilon);
    });

    AT_DISPATCH_FLOATING_TYPES(input_type, "icn_cuda", [&] {
        batch_norm_transform_input_kernel<scalar_t><<<blocks_trans, threads_trans>>>(
            input_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, size_t>(), 
            output_reshaped.packed_accessor<scalar_t, 3, RestrictPtrTraits, size_t>(), 
            save_mean_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            save_invstd_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            weight_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            bias_.packed_accessor<scalar_t, 1, RestrictPtrTraits, size_t>(), 
            epsilon);
    });
    return {output_reshaped.view(input.sizes()), save_mean_, save_invstd_};
}