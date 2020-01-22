/*
Follow pytorch official tutorial.
1. write a c++ file, 
define functions (called from python), 
and binds them to python with pybind 11,
also defined in cuda files

2. 

*/

#include <torch/extension.h>

#include <vector>

using namespace at;

namespace {
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
}

// CUDA forward declarations

std::vector<torch::Tensor> bcn_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    double minl,
    double epsilon);

std::vector<torch::Tensor> bcn_cuda_backward(
    torch::Tensor grad,
    const torch::Tensor X,
    const torch::Tensor weights,
    const torch::Tensor x_mean,
    const torch::Tensor inv_std,
    double clip,
    std::array<bool,3> grad_input_mask
    );

std::vector<torch::Tensor> icn_cuda_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    double minl,
    double epsilon);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> bcn_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    double minl,
    double epsilon) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  // CHECK_INPUT(minl);
  // CHECK_INPUT(epsilon);

  // return bcn_cuda_forward(input, weights, bias, minl, epsilon);
  return bcn_cuda_forward(input, weights, bias, minl, epsilon);
}

std::vector<torch::Tensor> bcn_backward(
    torch::Tensor grad,
    const torch::Tensor X,
    const torch::Tensor weights,
    const torch::Tensor x_mean,
    const torch::Tensor inv_std,
    double clip) {
  CHECK_INPUT(grad);
  CHECK_INPUT(X);
  // CHECK_INPUT(clip);
  CHECK_INPUT(inv_std);
  CHECK_INPUT(x_mean);
  CHECK_INPUT(weights);

  std::array<bool,3> grad_input_mask = {true,true,true};
  return bcn_cuda_backward(
    grad,
    X,
    weights,
    x_mean,
    inv_std,
    clip,
    grad_input_mask);
}

std::vector<torch::Tensor> icn_forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    double minl,
    double epsilon) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    CHECK_INPUT(bias);

    std::vector<int64_t> shape = input.sizes().vec();
    int64_t b = input.size(0);
    int64_t c = input.size(1);
    shape[1] = b * c;
    shape[0] = 1;

    torch::Tensor weight_ = repeat_if_defined(weights, b);
    torch::Tensor bias_ = repeat_if_defined(bias, b);
    auto input_reshaped = input.contiguous().view(shape);
    
    auto result = bcn_cuda_forward(input_reshaped, weight_, bias_, minl, epsilon);
    result[0] = result[0].view(input.sizes());
    return result;
}

std::vector<torch::Tensor> icn_backward(
    torch::Tensor grad,
    torch::Tensor X,
    torch::Tensor weights,
    torch::Tensor x_mean,
    torch::Tensor inv_std,
    double clip) {
    CHECK_INPUT(grad);
    CHECK_INPUT(X);
    CHECK_INPUT(inv_std);
    CHECK_INPUT(x_mean);
    CHECK_INPUT(weights);

    std::array<bool,3> grad_input_mask = {true,true,true};
    std::vector<int64_t> shape = X.sizes().vec();
    int64_t b = X.size(0);
    int64_t c = X.size(1);
    shape[1] = b * c;
    shape[0] = 1;

    torch::Tensor weight_ = repeat_if_defined(weights, b);
    auto X_reshaped = X.contiguous().view(shape);

    auto ans = bcn_cuda_backward(grad, X_reshaped, weight_, x_mean, inv_std, clip, grad_input_mask);    // ginp, gw, gb
    ans[0] = ans[0].view_as(X);
    ans[1] = ans[1].view({b,c}).sum(0);
    ans[2] = ans[2].view({b,c}).sum(0);
    return ans;
}
// std::string icn_backward_doc = "ICN backward (CUDA)\nargs:   grad_out, input, weight, save_mean, save_invstd, minl\nreturn: grad_input, grad_weight, grad_bias\n";

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.doc() = "X Condition Norm in CUDA";

m.def("batch_forward", &bcn_forward, \
"BCN forward (CUDA)\n\
args:   input, weight, bias, minl, epsilon\n\
return: output, save_mean, save_invstd\n");

m.def("batch_backward", &bcn_backward, \
"BCN backward (CUDA)\n\
args:   grad_out, input, weight, save_mean, save_invstd, minl\n\
return: grad_input, grad_weight, grad_bias\n");

m.def("instance_forward", &icn_forward, \
"ICN forward (CUDA)\n\
args:   input, weight, bias, minl, epsilon\n\
return: output, save_mean, save_invstd\n");

m.def("instance_backward", &icn_backward, \
"ICN backward (CUDA)\n\
args:   grad_out, input, weight, save_mean, save_invstd, minl\n\
return: grad_input, grad_weight, grad_bias\n");
}
