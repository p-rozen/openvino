// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace GNAPluginNS {

#define SCALE_FACTOR_GUARDBAND 1.25f

class Quantization {
public:
    void SetScale(float s) {
        scale = s;
        scale_set = true;
    }
    float GetScale() const {
        return scale;
    }
    bool IsScaleSet() const {
        return scale_set;
    }
    void SetLevels(int32_t l) {
        levels = l;
    }
    int32_t GetLevels() const {
        return levels;
    }
    void SetMinValues(const std::vector<float> &min) {
        min_values.clear();
        min_values.insert(min_values.end(), min.begin(), min.end());
    }
    const std::vector<float>& GetMinValues() const {
        return min_values;
    }
    void SetMaxValues(const std::vector<float>& max) {
        max_values.clear();
        max_values.insert(max_values.end(), max.begin(), max.end());
    }
    const std::vector<float>& GetMaxValues() const {
        return max_values;
    }

    void SetAgregatedDynamicRange(float dyn_range) {
        agg_dyn_range_ = dyn_range;
        agg_dyn_range_set_ = true;
    }
    float GetAgregatedDynamicRange() const {
        return agg_dyn_range_;
    }
    bool IsAgregatedDynamicRangeSet() const {
        return agg_dyn_range_set_;
    }

    void SetDynamicRange(float dyn_range) {
        dyn_range_ = dyn_range;
        dyn_range_set_ = true;
    }
    float GetDynamicRange() const {
        return dyn_range_;
    }
    bool IsDynamicRangeSet() const {
        return dyn_range_set_;
    }

    float CalculateScaleFactorBasedOnDynamicRange(float default_scale)
    {
        if (agg_dyn_range_set_) {
            float sf = 32768.0f / ceil(agg_dyn_range_ * SCALE_FACTOR_GUARDBAND);
            return sf;
        } else {
            return scale_set ? scale : default_scale;
        }
    }
private:
    float scale = 1.0f;
    bool scale_set = false;
    int32_t levels = 0;
    std::vector<float> min_values;
    std::vector<float> max_values;

    float dyn_range_ = 0.0f;
    bool dyn_range_set_ = false;
    float agg_dyn_range_ = 0.0f;
    bool agg_dyn_range_set_ = false;

};

struct QuantizedLayerParams {
    Quantization _src_quant;
    Quantization _dst_quant;

    // deprecate this
    Quantization _weights_quant;
    bool _weights_quantized = false;
    Quantization _bias_quant;
    float _o_shift = 0.0f;
    float _b_shift = 0.0f;
};

}  // namespace GNAPluginNS
