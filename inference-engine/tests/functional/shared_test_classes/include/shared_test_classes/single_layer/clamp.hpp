// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {

using ClampParamsTuple = typename std::tuple<
        std::vector<std::vector<size_t>>, //input shapes
        InferenceEngine::Precision,       //Network precision
        InferenceEngine::Precision,       // Input precision
        InferenceEngine::Precision,       // Output precision
        InferenceEngine::Layout,          // Input layout
        InferenceEngine::Layout,          // Output layout
        std::string,                      // Device name
        std::vector<float>>;              // max/min clamp

class ClampLayerTest:
        public testing::WithParamInterface<ClampParamsTuple>,
        virtual public LayerTestsUtils::LayerTestsCommon{
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ClampParamsTuple> &obj);
protected:
    void SetUp() override;
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
};
}  // namespace LayerTestsDefinitions
