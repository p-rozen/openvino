// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/clamp.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{1, 8}},
        {{2, 16}},
        {{3, 32}},
        {{4, 64}},
        {{5, 128}},
        {{6, 256}},
        {{7, 512}},
        {{8, 1024}}
};

std::vector<std::vector<float >> clamp_min_max = {
        {-16.0f, 0.0f},
        {-1.0, 1.0},
        {-5.0f, 5.0f},
        {-10, 10},
        {-15, 15},
        {0, 16},
};

std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                         InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_CASE_P(smoke_power, ClampLayerTest,
        ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(clamp_min_max)),
        ClampLayerTest::getTestCaseName);
}  // namespace
