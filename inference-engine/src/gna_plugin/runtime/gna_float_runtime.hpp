// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#define GEN_STATS
#include <backend/am_intel_dnn.hpp>

#ifdef GEN_STATS
#include "backend/stats_dao.hpp"
#endif

namespace GNAPluginNS {
namespace runtime {
/**
 * @brief floating runtime for gna-plugin, in most case it uses same gna-primitives description as integer runtime, but execute them on CPU
 */
class FP {
    std::shared_ptr<backend::AMIntelDNN> dnn;
 public:
#ifdef GEN_STATS
    FP(std::shared_ptr<backend::AMIntelDNN> dnn) : dnn(dnn), stats_(nullptr) {
#else
     FP(std::shared_ptr<backend::AMIntelDNN> dnn) : dnn(dnn) {
#endif
    }
    virtual ~FP() {
#ifdef GEN_STATS
        if (stats_) {
            stats_->Serialize("layer_statistics.txt");
            delete stats_;
        }
#endif
    }
    virtual void infer();

    /**
     * atomic operations for floating inference
     */
    static void ApplyAffineTransform(intel_dnn_component_t *component, uint32_t *list, uint32_t listsize);
    static void ApplyDiagonalTransform(intel_dnn_component_t *component);
    static void ApplyRecurrentTransform(intel_dnn_component_t *component, uint32_t row, void *ptr_feedbacks);
    static void ApplyConvolutional1DTransform(intel_dnn_component_t *component);
    static void ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                              intel_dnn_number_type_t number_type,
                                              uint32_t listsize);
    static void ApplyPiecewiseLinearTransform(intel_dnn_component_t *component,
                                              intel_dnn_number_type_t number_type,
                                              uint32_t listsize,
                                              uint32_t num_row);
    static void ApplyMaxPoolTransform(intel_dnn_component_t *component, intel_dnn_number_type_t number_type);
    static void ApplyTranspose(intel_dnn_component_t *component);
    static void ApplyCopy(intel_dnn_component_t *component);

protected:
#ifdef GEN_STATS
    StatisticsDao *stats_;
#endif

};

}  // namespace runtime

}  // namespace GNAPluginNS
