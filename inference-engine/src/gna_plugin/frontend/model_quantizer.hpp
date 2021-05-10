// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>
#include <string>

#include <legacy/layer_transform.hpp>
#include "gna_graph_tools.hpp"
#include <legacy/details/ie_cnn_network_tools.h>
#include "layer_quantizer.hpp"
#include "scale_factor_calc.hpp"
#include "weights_converter.hpp"
#include "backend/stats_dao.hpp"

namespace GNAPluginNS {

/**
 * Quantize entire cnn - network
 * @tparam T - type trait for weights and biases
 */
template<class T>
class ModelQuantizer {
 public:
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, float scaleFactor) const {
        return quantize(model, [](const InferenceEngine::CNNNetwork &, bool runBeforeCopy){}, std::vector<float>({scaleFactor}));
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, const PreQuantisationCb &cb, float scaleFactor) const {
        return quantize(model, cb, std::vector<float>({scaleFactor}));
    }

    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, std::vector<float> scaleFactor) const {
        return quantize(model, [](InferenceEngine::CNNNetwork &, bool runBeforeCopy){}, scaleFactor);
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, const PreQuantisationCb &cb, std::vector<float> scaleFactor) const {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            auto newLayer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
            transformLayer(newLayer, WeightsConverter());
            return newLayer;
        };
        InferenceEngine::CNNNetwork copiedNet = InferenceEngine::CNNNetCopy(model);
        cb(copiedNet, true);

        copiedNet = InferenceEngine::CNNNetCopy(copiedNet, visitor);

        // allow client code to access copied topology, to avoid copies if user would like to chain quantisation with
        // another preprocessing
        cb(copiedNet, false);

        if (scaleFactor.empty()) {
            THROW_GNA_EXCEPTION << "Scale factor is empty";
        }

        LayersQuantizer<T> lc(*scaleFactor.begin());
        auto sortedNewNet = InferenceEngine::details::CNNNetSortTopologically(copiedNet);
        gnalog() << "Sorted layers: " << std::endl;
        for (auto &&layer : sortedNewNet) {
            gnalog() << layer->name << std::endl;
        }
        /// filling scale factors for input layers, memory layers will have scaleFactor of 1.0 by default
        InferenceEngine::InputsDataMap dm = copiedNet.getInputsInfo();
        int scaleIndex = 0;
        for (auto &&inputData : dm) {
            auto inputLayer = getCreatorLayer(inputData.second->getInputData()).lock();
            auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            if (scaleFactor.size() <= scaleIndex) {
                THROW_GNA_EXCEPTION << "Scale factors are not set for some of the inputs";
            }
            IE_ASSERT(quantData != nullptr);
            quantData->_src_quant.SetScale(scaleFactor[scaleIndex]);
            scaleIndex++;
        }
        const float MIN_DYNAMIC_RANGE = 1e-20f;
        //StatisticsDao* stats = nullptr;

        StatisticsDao* stats = StatisticsDao::Deserialize("layer_statistics.txt");
        int index = 0;
        for (auto& layer : sortedNewNet) {
            auto curr_quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
            if (curr_quant_data) {
                if (layer->insData.size()) {
                    curr_quant_data->_dst_quant.SetAgregatedDynamicRange(5.0f / SCALE_FACTOR_GUARDBAND);
                    curr_quant_data->_src_quant.SetAgregatedDynamicRange(5.0f / SCALE_FACTOR_GUARDBAND);
                }
            }
            index++;
        }

        if (stats) {
            for (auto& layer : sortedNewNet) {
                auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
                int layer_id = stats->GetLayerId(layer->name.c_str());
                if (quantData && layer_id >= 0) {
                    float min, max;
                    stats->GetMinMax(layer_id, StatisticsDao::stats_type_e::OUTPUT, min, max);
                    // if min > max - scale factor not set
                    if (max > min) {
                        float absmax = max > -min ? max : -min;
                        quantData->_dst_quant.SetDynamicRange(absmax);
                    }
                }
            }
            delete stats;
            bool finished = false;
            // overwrite dynamic range of input base on scale factor of input
            // this is done to unify scaling factor for all chunks of inputs
            for (auto& layer : sortedNewNet) {
                auto it = LayerNameToType.find(layer->type);
                if (it != LayerNameToType.end() && it->second == LayerType::Input)
                {
                    auto in_quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
                    in_quant_data->_src_quant.SetDynamicRange(MAX_VAL_2B_FEAT / in_quant_data->_src_quant.GetScale());
                    in_quant_data->_src_quant.SetAgregatedDynamicRange(MAX_VAL_2B_FEAT / in_quant_data->_src_quant.GetScale());
                    in_quant_data->_dst_quant.SetDynamicRange(MAX_VAL_2B_FEAT / in_quant_data->_src_quant.GetScale());
                    in_quant_data->_dst_quant.SetAgregatedDynamicRange(MAX_VAL_2B_FEAT / in_quant_data->_src_quant.GetScale());
                }
            }
            while (!finished) {
                finished = true;
                for (auto& layer : sortedNewNet) {
                    auto curr_quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
                    if (curr_quant_data) {
                        if (layer->insData.size()) {
                            float agg_dynamic_range = 0.0f;
                            for (auto in_layer_ptr : layer->insData) {
                                auto in_layer = getCreatorLayer(in_layer_ptr.lock()).lock();
                                auto in_quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(in_layer);
                                if (in_quant_data && (in_quant_data->_dst_quant.IsDynamicRangeSet() ||
                                    in_quant_data->_dst_quant.IsAgregatedDynamicRangeSet())) {
                                    // if this is first visit - we need to take dynamic range, which is not aggregated yet
                                    agg_dynamic_range = agg_dynamic_range > in_quant_data->_dst_quant.GetDynamicRange() ?
                                        agg_dynamic_range : in_quant_data->_dst_quant.GetDynamicRange();
                                    // and aggregated one too
                                    agg_dynamic_range = agg_dynamic_range > in_quant_data->_dst_quant.GetAgregatedDynamicRange() ?
                                        agg_dynamic_range : in_quant_data->_dst_quant.GetAgregatedDynamicRange();
                                }
                            }

                            if (!LayerInfo(layer).isChangingDynamicRange()) {
                                agg_dynamic_range = agg_dynamic_range > curr_quant_data->_src_quant.GetAgregatedDynamicRange() ?
                                    agg_dynamic_range : curr_quant_data->_src_quant.GetAgregatedDynamicRange();
                                agg_dynamic_range = agg_dynamic_range > curr_quant_data->_dst_quant.GetAgregatedDynamicRange() ?
                                    agg_dynamic_range : curr_quant_data->_dst_quant.GetAgregatedDynamicRange();
                                if (agg_dynamic_range > curr_quant_data->_src_quant.GetAgregatedDynamicRange() ||
                                    agg_dynamic_range > curr_quant_data->_dst_quant.GetAgregatedDynamicRange())
                                {
                                    if (!curr_quant_data->_src_quant.IsAgregatedDynamicRangeSet())
                                        finished = false;
                                    curr_quant_data->_src_quant.SetAgregatedDynamicRange(agg_dynamic_range);
                                    if (!curr_quant_data->_dst_quant.IsAgregatedDynamicRangeSet())
                                        finished = false;
                                    curr_quant_data->_dst_quant.SetAgregatedDynamicRange(agg_dynamic_range);
                                }
                            }
                            else {
                                if (curr_quant_data->_src_quant.GetAgregatedDynamicRange() < agg_dynamic_range)
                                    finished = false;
                                curr_quant_data->_src_quant.SetAgregatedDynamicRange(agg_dynamic_range);
                            }
                            if (agg_dynamic_range > MIN_DYNAMIC_RANGE) {
                                for (auto& in_layer_ptr : layer->insData)
                                {
                                    auto in_layer = getCreatorLayer(in_layer_ptr.lock()).lock();
                                    auto in_quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(in_layer);
                                    if (in_quant_data)
                                    {
                                        if (in_quant_data->_dst_quant.GetAgregatedDynamicRange() < agg_dynamic_range)
                                        {
                                            in_quant_data->_dst_quant.SetAgregatedDynamicRange(agg_dynamic_range);
                                            finished = false;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        propagateScaleFactor(sortedNewNet, T::mandatory().getWeightsPrecision().size());

        for (auto &layer : sortedNewNet)
        {
            auto curr_quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
            printf("Dynamic range of layer: %s Input: %.5f Output: %.5f Scale factors: %.5f/%.5f/%.5f\n", layer->name.c_str(),
                curr_quant_data->_src_quant.GetAgregatedDynamicRange(),
                curr_quant_data->_dst_quant.GetAgregatedDynamicRange(),
                curr_quant_data->_src_quant.GetScale(),
                curr_quant_data->_weights_quant.GetScale(),
                curr_quant_data->_dst_quant.GetScale());
        }

        // sorted order gives possibility for propagate quantisation along depended layers
        for (auto &&layer : sortedNewNet) {
            transformLayer(layer, lc);
        }

        return copiedNet;
    }

 private :
    void propagateScaleFactor(std::vector<InferenceEngine::CNNLayerPtr> & net, int weightsBytesSize) const {
        ScaleFactorCalculator sf(net, weightsBytesSize);

        while (!sf.allLayersProcessed()) {
            for (auto &&layer : sf.getStartLayers()) {
                transformLayer(layer, sf);
                // transforming until we reached cases where output scale updated due to situation in downstream layer
                if (sf.needToRestart()) {
                    break;
                }
            }
        }
    }
};
}  // namespace GNAPluginNS
