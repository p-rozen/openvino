// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "strided_slice_inst.h"
#include "primitive_gpu_base.h"
#include "implementation_map.h"
#include "kernel_selector_helper.h"
#include "strided_slice/strided_slice_kernel_ref.h"
#include "strided_slice/strided_slice_kernel_selector.h"
#include "error_handler.h"
#include "data_inst.h"
#include <vector>

using namespace cldnn;

namespace cldnn {
namespace gpu {

struct strided_slice_gpu : typed_primitive_gpu_impl<strided_slice> {
    using parent = typed_primitive_gpu_impl<strided_slice>;
    using parent::parent;

public:
    static primitive_impl* create(const strided_slice_node& arg) {
        auto params = get_default_params<kernel_selector::strided_slice_params>(arg);
        auto op_params = get_default_optional_params<kernel_selector::strided_slice_optional_params>(arg.get_program());
        const size_t dims_num = params.inputs[0].Dimentions();

        // Getting data from constant inputs. There are 3 args: Begin, End, Stride
        for (size_t i = 1; i < arg.get_dependencies().size(); ++i) {
            auto& input = arg.get_dependency(i).as<data>();
            auto& mem = input.get_attached_memory();
            std::vector<int32_t> sizes;
            if (input.get_output_layout().data_type == cldnn::data_types::i64) {
                int64_t* data = static_cast<int64_t*>(mem.lock());
                std::vector<int64_t> sizes_i64 = std::vector<int64_t>(data, data + input.get_output_layout().count());
                sizes.resize(sizes_i64.size());
                for (size_t j = 0; j < sizes.size(); j++)
                    sizes[j] = static_cast<int32_t>(sizes_i64[j]);
            } else {
                int32_t* data = static_cast<int32_t*>(mem.lock());
                sizes = std::vector<int32_t>(data, data + input.get_output_layout().count());
            }
            pad_vector_to_size(sizes, dims_num, i != 1);  // for "begin" completion used 0 value, for other - 1
            params.striding_params.push_back(sizes);
            mem.unlock();
        }

        params.end_mask = arg.get_primitive()->end_mask;
        pad_vector_to_size(params.end_mask, dims_num, 1);
        params.begin_mask = arg.get_primitive()->begin_mask;
        pad_vector_to_size(params.begin_mask, dims_num, 1);
        params.new_axis_mask = arg.get_primitive()->new_axis_mask;
        params.shrink_axis_mask = arg.get_primitive()->shrink_axis_mask;
        pad_vector_to_size(params.shrink_axis_mask, dims_num, 0);

        std::vector<size_t> logical_dims = params.inputs[0].LogicalDims();
        std::reverse(logical_dims.begin(), logical_dims.end());  // get dims in bfyx order
        std::vector<int32_t> out_shape;
        for (const auto& dim : logical_dims)
            out_shape.push_back(static_cast<int32_t>(dim));
        // If the ith bit of begin_mask is not set, begin[i] is ignored and the range of the appropriate dimension starts from 0.
        vector_assign_if_not_mask(params.striding_params[0], 0, params.begin_mask);
        // If the ith bit of end_mask is not set, end[i] is ignored and the fullest possible range in that dimension is used
        // instead.
        vector_assign_if_not_mask(params.striding_params[1], out_shape, params.end_mask);
        for (size_t dim = 0; dim < params.striding_params[2].size(); dim++) {
            if (params.striding_params[0][dim] < 0)
                params.striding_params[0][dim] = std::max(out_shape[dim] + params.striding_params[0][dim], (int32_t)0);
            if (params.striding_params[1][dim] < 0)
                params.striding_params[1][dim] = std::max(out_shape[dim] + params.striding_params[1][dim], (int32_t)0);

            params.striding_params[0][dim] = std::min(params.striding_params[0][dim], out_shape[dim]);
            params.striding_params[1][dim] = std::min(params.striding_params[1][dim], out_shape[dim]);

            auto& begin = params.striding_params[0][dim];
            auto& end = params.striding_params[1][dim];
            auto& stride = params.striding_params[2][dim];
            bool is_reverse = stride < 0;
            // If begin > end && is_reverse, then we don't need to adjust begin/end values, the kernel will process it correctly
            // If begin <= end, then we swap begin/end values and subtruct 1 from each of them
            // E.g. out_shape[dim] = 100; begin=0; end=100; stride=-1
            // swap: begin=100; end=0;
            // sub: begin=99; end=-1;
            // So the kernel will put the slices [99, 0] in reversed order as expected.
            if (is_reverse && begin <= end) {
                std::swap(begin, end);
                begin--;
                end--;
            }
        }

        auto& kernel_selector = kernel_selector::strided_slice_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(params, op_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto strided_slice = new strided_slice_gpu(arg, best_kernels[0]);

        return strided_slice;
    }
};

namespace detail {

attach_strided_slice_gpu::attach_strided_slice_gpu() {
    auto val_fw = strided_slice_gpu::create;
    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), val_fw);
    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), val_fw);
    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfyx), val_fw);
    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfyx), val_fw);

    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfzyx), val_fw);
    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfzyx), val_fw);
    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::i32, format::bfzyx), val_fw);
    implementation_map<strided_slice>::add(std::make_tuple(engine_types::ocl, data_types::i64, format::bfzyx), val_fw);
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
