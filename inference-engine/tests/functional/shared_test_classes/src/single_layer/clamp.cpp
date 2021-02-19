// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/clamp.hpp"

namespace LayerTestsDefinitions {
std::string ClampLayerTest::getTestCaseName(const testing::TestParamInfo<ClampParamsTuple> &obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::vector<float> power;
    std::tie(inputShapes, netPrecision, inPrc, outPrc, inLayout, outLayout, targetName, power) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    results << "Power=" << CommonTestUtils::vec2str(power) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "inPRC=" << inPrc.name() << "_";
    results << "outPRC=" << outPrc.name() << "_";
    results << "inL=" << inLayout << "_";
    results << "outL=" << outLayout << "_";
    results << "trgDev=" << targetName << "_";
    return results.str();
}

void ClampLayerTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::vector<float> min_max_clamp;
    std::tie(inputShapes, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice, min_max_clamp) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto paramsIn = ngraph::builder::makeParams(ngPrc, {inputShapes[0]});
    auto clamp = std::make_shared<ngraph::opset1::Clamp>(paramsIn[0], min_max_clamp[0], min_max_clamp[1]);

    function = std::make_shared<ngraph::Function>(clamp, paramsIn, "clamp");
}

InferenceEngine::Blob::Ptr ClampLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 17, -17);
}
} // namespace LayerTestsDefinitions
