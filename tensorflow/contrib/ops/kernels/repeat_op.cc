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

template <typename T>
void RepeatCPUImplSingleThreaded(OpKernelContext* ctx,
  const Tensor& input,
                   const typename TTypes<int32>::ConstFlat& repeats_flat,
                   int axis, Tensor* output) {
  auto input_flat = input.flat<T>();
  auto output_flat = output->flat<T>();
    
  // a batch is inner axes > axis
  size_t batch_size = 1;
  int32 dims = input.shape().dims();
  for (int32 i = axis + 1; i < dims; ++i) {
    batch_size *= input.shape().dim_size(i);
  }
  int64 num_batch = input_flat.size() / batch_size;
  
  const T* in = input_flat.data();
  T* out = output_flat.data();
  
  // copy an in_batch to its out_batches
  auto handle_batch = [&in, batch_size, &out](int32 repeat) {
    for (int64 j = 0; j < repeat; ++j) {
      std::copy(in, in + batch_size, out);
      out += batch_size;
    }
    in += batch_size;
    return;
  };
  
  if (repeats_flat.size() == 1) {
    for (int64 i = 0; i < num_batch; ++i) {
      handle_batch(repeats_flat(0));
    }
  } else {
    for (int64 i = 0; i < num_batch; ++i) {
      handle_batch(repeats_flat(i % repeats_flat.size()));
    }
  }
}

template <typename T>
void RepeatCPUImplMultiThreaded(OpKernelContext* ctx, DeviceBase* d, const Tensor& input,
                     const typename TTypes<int32>::ConstFlat& repeats_flat,
                     int axis, int64 cost_per_unit, Tensor* output) {
  auto input_flat = input.flat<T>();
  auto output_flat = output->flat<T>();
  
  // a batch is inner axes > axis
  // a group is inner axes >= axis
  int64 batch_size = 1;
  int32 dims = input.shape().dims();
  for (int32 i = axis + 1; i < dims; ++i) {
    batch_size *= input.shape().dim_size(i);
  }
  int64 group_pre_size = batch_size * input.shape().dim_size(axis);
  int64 group_size = batch_size * output->shape().dim_size(axis);
  
  auto worker_threads = d->tensorflow_cpu_worker_threads();
  int num_threads = std::min(4, worker_threads->num_threads);
  // strings define a different amount of work (generally much more) compared
  // with standard POD, so we parallelize differently.
  if (!std::is_same<T, string>::value) {
    num_threads =
        static_cast<int>(std::min<int64>(num_threads, output_flat.size() / 4096));
  }
  
  if (num_threads == 0) {
    RepeatCPUImplSingleThreaded<T>(input, repeats_flat, axis, output);
    return;
  }
  
  auto work = [input_flat, repeats_flat, axis,
               batch_size, group_pre_size, group_size, &output_flat](
      int64 out_begin_index, int64 out_end_index) {
    const T* in = input_flat.data();
    T* out = output_flat.data();
    T* out_start = out + out_begin_index;
    T* out_end = out + out_end_index;
    
    // handle partial group at start
    int64 skip_group = out_begin_index / group_size;
    in += skip_group * group_pre_size;
    out += skip_group * group_size;
    
    if (out_begin_index % group_size != 0) {
      for (int64 j = 0; j < repeats_flat.size(); ++j) {
        for (int64 k = 0; k < repeats_flat(j); ++k) {
          if (out + batch_size <= out_start) {
            out += batch_size;
            continue;
          }
          
          int64 offset = out_start - out;
          offset = offset>0 ? offset : 0;
          if (out + batch_size > out_end) {
            std::copy(in + offset, in + (out_end-out), out + offset);
            return;
          }
          std::copy(in + offset, in + batch_size, out + offset);
          
          out += batch_size;
        }
        in += batch_size;
      }
    }
    
    // handle remaining data
    int64 group_to_cpy = (out_end-out) / group_size + 1;
    for (int64 i = 0; i < group_to_cpy; ++i) {
      for (int64 j = 0; j < repeats_flat.size(); ++j) {
        for (int64 k = 0; k < repeats_flat(j); ++k) {
          if (out + batch_size > out_end) {
            std::copy(in, in + (out_end-out), out);
            return;
          }
          std::copy(in, in + batch_size, out);
          out += batch_size;
        }
        in += batch_size;
      }
    }
  };

  Shard(worker_threads->num_threads, worker_threads->workers, output_flat.size(),
        cost_per_unit, work);
}


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
