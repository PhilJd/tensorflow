/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/contrib/repeat/kernels/repeat_op.h"
#include "tensorflow/core/util/work_sharder.h"


// used in cpu implementation sharded mode
const int kCostPerUnit = 10000;

typedef Eigen::ThreadPoolDevice CPUDevice;
#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;
#endif // GOOGLE_CUDA

namespace tensorflow{



template <typename Device, typename T, typename DeviceRepeatFunctor>
class RepeatOp : public OpKernel {
 public:
  explicit RepeatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }
  
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& repeats = context->input(1);
    const int input_rank = (input.dims() == 0) ? 1 : input.dims();
    const int32 axis = (axis_ >= 0) ? axis_ : axis_ + input_rank;
    
    OP_REQUIRES(context, TensorShapeUtils::IsVector(repeats.shape()) ||
                         TensorShapeUtils::IsScalar(repeats.shape()),
                errors::InvalidArgument("`repeats` expects a scalar or a 1-D vector."));
    OP_REQUIRES(context, FastBoundsCheck(axis, input_rank),
                errors::InvalidArgument(
                    "Expected -", input_rank, " <= `axis` < ", input_rank));
    OP_REQUIRES(context, repeats.NumElements() == input.dim_size(axis) ||
                         repeats.NumElements() == 1,
                errors::InvalidArgument(
                    "Expected `repeats` argument to be a vector of length ",
                    input.dim_size(axis_), " or 1, but got length ",
                    repeats.NumElements()));
    // compute the output shape
    const auto repeats_flat = repeats.flat<int32>();
    bool input_is_scalar = input.dims() == 0;
    TensorShape output_shape = input_is_scalar ? TensorShape({1})
                                               : input.shape();
    int old_dim = input_is_scalar ? 1 : input.shape().dim_size(axis);
    repeat_dim_size = repeats.NumElements() == 1 ? repeats_flat(0) * old_dim
                                                 : repeats_flat.sum();
    output_shape.set_dim(axis, repeat_dim_size);
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    repeat_functor_(context, input, repeats_flat, axis, output);
  }
  
 private:
  int32 axis_;
  DeviceRepeatFunctor repeat_functor_;
  
};


#define REGISTER_KERNEL(type)                             \
  REGISTER_KERNEL_BUILDER(                                \
      Name("Repeat")                                      \
      .Device(DEVICE_CPU)                                 \
      .TypeConstraint<type>("T"),                         \
      RepeatOp<CPUDevice, type, RepeatCPUFunctor<type> >)

TF_CALL_ALL_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#if GOOGLE_CUDA
  
#define REGISTER_KERNEL_GPU(type)                         \
  REGISTER_KERNEL_BUILDER(                                \
      Name("Repeat")                                      \
      .Device(DEVICE_GPU)                                 \
      .TypeConstraint<type>("T")                          \
      .HostMemory("repeats"),                             \
      RepeatOp<GPUDevice, type, RepeatGPUFunctor<type> >)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_KERNEL_GPU);

#undef REGISTER_KERNEL_GPU

#endif // GOOGLE_CUDA

} //end namespace tensorflow
