// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/group_convolution_decomposition.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

using namespace ngraph;
using namespace op;

NGRAPH_RTTI_DEFINITION(ngraph::pass::GroupConvolutionDecomposition, "GroupConvolutionDecomposition", 0);

std::shared_ptr<opset1::StridedSlice> FlatCrop(Output<Node> input, size_t offset, size_t size)
{
    auto shape = input.get_shape();
    if (shape.size() == 1) {
        return std::make_shared<ngraph::opset1::StridedSlice>(
            input, // data
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { offset }), // begin slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { offset+size }), // end slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 1 }, { 1 }), // strides
            std::vector<int64_t>{0}, // begin mask
            std::vector<int64_t>{0} // end mask
        );
    } else if (shape.size() == 2) {
        return std::make_shared<ngraph::opset1::StridedSlice>(
            input, // data
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)0, offset }), // begin sice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)0, offset + size }), // end slice index
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ 2 }, { (size_t)1, (size_t)1 }), // strides
            std::vector<int64_t>{1,0}, // begin mask
            std::vector<int64_t>{1,0} // end mask
        );
    }
    return nullptr;
}

std::shared_ptr<opset1::Reshape> Flatten(Output<Node> input)
{
    auto shape = input.get_shape();
    size_t size = 1;
    for (auto d : shape)
        size *= d;
    return std::make_shared<opset1::Reshape>(input, Shape{ (size_t)1, size });
}

ngraph::pass::GroupConvolutionDecomposition::GroupConvolutionDecomposition() {
    MATCHER_SCOPE(GroupConvolutionDecomposition);
    auto gc = pattern::wrap_type<opset1::GroupConvolution>({
        pattern::any_input(pattern::has_static_rank()),
        pattern::any_input(pattern::has_static_shape())
    });

    ngraph::matcher_pass_callback callback = [this](ngraph::pattern::Matcher &m) {
        auto m_gc = m.get_match_root();

        auto m_gc_v1 = std::dynamic_pointer_cast<opset1::GroupConvolution>(m_gc);
        if (!m_gc_v1) {
            return false;
        }
        Output<Node>& m_input = m_gc_v1->input_value(0);
        Output<Node>& m_filters = m_gc_v1->input_value(1);
        
        Strides m_strides = m_gc_v1->get_strides();
        CoordinateDiff m_pads_begin = m_gc_v1->get_pads_begin();
        CoordinateDiff m_pads_end = m_gc_v1->get_pads_end();
        Strides m_dilations = m_gc_v1->get_dilations();
        PadType m_auto_pad = m_gc_v1->get_auto_pad();

        const auto& input_type = m_input.get_element_type();
        auto input_shape = m_input.get_shape();
        auto kernel_shape = m_filters.get_shape();
        auto output_shape = m_gc_v1->get_output_shape(0);

        int32_t kernel_count;
        int32_t kernel_height;
        int32_t kernel_width;

        int32_t kernel_depth;
        int32_t stride_x;
        int32_t stride_y;

        int32_t input_width;
        int32_t input_height;
        int32_t input_channel_count;

        int32_t output_channel_count = input_channel_count;
        int32_t output_width;
        int32_t output_height;

        if (input_shape.size() == 4)
        {
            //NCHW - batch size 1
            if (1 != input_shape[0])
            {
                return false;
            }
            input_channel_count = input_shape[1];
            input_height = input_shape[2];
            input_width = input_shape[3];
        } else if (input_shape.size() == 3) {
            //NCW - batch size 1
            if (1 != input_shape[0])
            {
                return false;
            }
            input_channel_count = input_shape[1];
            input_height = 1;
            input_width = input_shape[2];

            return false;
        }

        if (kernel_shape.size() == 4)
        {
            //NCHW
            kernel_count = kernel_shape[0];
            kernel_depth = kernel_shape[1];
            kernel_height = input_shape[2];
            kernel_width = input_shape[3];
        } else if (kernel_shape.size() == 3) {
            //NCW
            kernel_count = kernel_shape[0];
            kernel_depth = kernel_shape[1];
            kernel_height = 1;
            kernel_width = input_shape[2];
        } else {
            return false;
        }

        // check if convolution definition is valid
        if (kernel_count * kernel_depth != input_channel_count) {
            return false;
        }

        // transpose kernels
        auto constant = std::dynamic_pointer_cast<opset1::Constant>(m_filters.get_node_shared_ptr());
        std::shared_ptr<opset1::Constant> new_kernels_const = nullptr;
        if (constant && constant->get_element_type() == ngraph::element::f32) {
            auto nhw_kernels_data = constant->get_data_ptr<float>();
            std::vector<float> hwn_kernels_data;
            hwn_kernels_data.resize(kernel_count * kernel_height * kernel_width);

            for (int h = 0; h < kernel_height; h++) {
                for (int w = 0; w < kernel_width; w++) {
                    for (int k = 0; k < kernel_count; k++) {
                        hwn_kernels_data[h * kernel_width * kernel_count + w * kernel_count + k] =
                            nhw_kernels_data[k * kernel_height * kernel_width + h * kernel_width + w];
                    }
                }
            }
            new_kernels_const = std::make_shared<opset1::Constant>(ngraph::element::f32,
                kernel_shape.size() == 4 ? Shape{ (size_t)1, (size_t)kernel_height, (size_t)kernel_width, (size_t)kernel_count } :
                    Shape{ (size_t)1, (size_t)kernel_width, (size_t)kernel_count },
                nhw_kernels_data);
            ngraph::copy_runtime_info(m_gc_v1, new_kernels_const);
        } else {
            return false;
        }

        if (output_shape.size() == 4)
        {
            //NCHW - kernel depth must be 1 - we do not support grouping > 1
            if (1 != output_shape[0]) {
                return false;
            }
            output_channel_count = output_shape[1];
            output_height = output_shape[2];
            output_width = output_shape[3];
        } else if (input_shape.size() == 3) {
            //NCW - batch size 1
            if (1 != output_shape[0]) {
                return false;
            }
            output_channel_count = output_shape[1];
            output_height = 1;
            output_width = output_shape[2];

            return false;
        }
        //TODO: check m_auto_pad
        int32_t pads_begin_x = m_pads_begin.size() > 0 ? m_pads_begin[0] : 0;
        int32_t pads_end_x = m_pads_end.size() > 0 ? m_pads_end[0] : 0;

        int32_t pads_begin_y = m_pads_begin.size() > 1 ? m_pads_begin[1] : 0;
        int32_t pads_end_y = m_pads_end.size() > 1 ? m_pads_end[1] : 0;

        int32_t dilation_x = m_dilations.size() > 0 ? m_dilations[0] : 1;
        int32_t dilation_y = m_dilations.size() > 1 ? m_dilations[1] : 1;

        auto flat_input = Flatten(m_input);
        auto flat_kernels = Flatten(new_kernels_const);

        //TODO: add 0 if padding > kernel size*dilation
        if (kernel_depth == 1 &&
            pads_begin_x < kernel_width * dilation_x && pads_end_x < kernel_width * dilation_x &&
            pads_begin_y < kernel_height * dilation_y && pads_end_y < kernel_height * dilation_y)
        {
            //TODO: add transpose NCHW => NHWC
            int32_t real_output_height = 0;
            NodeVector concat_inputs;

            for (int32_t y = -pads_begin_y, oy = 0; oy < output_height; y += stride_y, oy++)
            {
                for (int32_t x = -pads_begin_x, ox = 0; ox < output_width; x += stride_x, ox++)
                {
                    int32_t kend_x = x + kernel_width * dilation_x;
                    kend_x = kend_x < input_width ? kend_x : input_width;

                    int32_t kend_y = y + kernel_height * dilation_y;
                    kend_y = kend_y < input_height ? kend_y : input_height;

                    bool first = true;
                    std::shared_ptr<ngraph::op::Op> conv_acc;

                    for (int32_t kpos_y = y, kernel_elem_id = 0; kpos_y < kend_y; kpos_y += dilation_y)
                    {
                        for (int32_t kpos_x = x, kernel_elem_id = 0; kpos_x < kend_x; kpos_x += dilation_x, kernel_elem_id++)
                        {
                            if (kpos_x >= 0 && kpos_y >= 0)
                            {
                                auto conv_input_w_ki = FlatCrop(flat_input, (kpos_y * input_width + kpos_x) * input_channel_count, input_channel_count);
                                auto conv_kernel_w_ki = FlatCrop(flat_kernels, kernel_elem_id * input_channel_count, input_channel_count);

                                // we must insert now
                                if (conv_acc)
                                {
                                    conv_acc = std::make_shared<opset1::Multiply>(conv_input_w_ki, conv_kernel_w_ki);
                                    ngraph::copy_runtime_info(m_gc_v1, {conv_input_w_ki, conv_kernel_w_ki, conv_acc });
                                }
                                else
                                {
                                    auto mul = std::make_shared<opset1::Multiply>(conv_input_w_ki, conv_kernel_w_ki);
                                    conv_acc = std::make_shared<opset1::Add>(mul, conv_acc);
                                    ngraph::copy_runtime_info(m_gc_v1, { conv_input_w_ki, conv_kernel_w_ki, mul, conv_acc });
                                }
                            }
                        }
                    }
                    // if padding greater than assumed exit greacefully
                    if (!conv_acc)
                        return false;
                    concat_inputs.push_back(conv_acc);
                }
            }
            auto concat = std::make_shared<opset1::Concat>(concat_inputs, 0);
            auto result = std::make_shared<opset1::Reshape>(concat, Shape{ (size_t)1, (size_t)output_height, (size_t)output_width, (size_t)output_channel_count }, false);
            //TODO: add transpose NHWC => NCHW
            ngraph::copy_runtime_info(m_gc_v1, { flat_input, flat_kernels, concat, result });
            ngraph::replace_node(m_gc_v1, result);
        }
        return true;
    };
    auto m = std::make_shared<ngraph::pattern::Matcher>(gc, matcher_name);
    this->register_matcher(m, callback);
}

