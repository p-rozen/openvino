// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <transformations_visibility.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace pass {

class TRANSFORMATIONS_API GroupConvolutionDecomposition;

}  // namespace pass
}  // namespace ngraph

/**
 * @ingroup ie_transformation_common_api
 * @brief GroupConvolutionDecomposition transformation decomposes Depth Wise Separable Convolution layer 
 * with inputs X, and kernels K, padding P, dilation D, Stride S
 * to sequence of element wise mul / add operations
 * Math is described on https://docs.openvinotoolkit.org/latest/openvino_docs_ops_convolution_GroupConvolution_1.html
  * Limitations:
 *    - number of elements in kernel multiplication of 32
 */

class ngraph::pass::GroupConvolutionDecomposition: public ngraph::pass::MatcherPass {
public:
    NGRAPH_RTTI_DECLARATION;
    GroupConvolutionDecomposition();
};
