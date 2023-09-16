#ifndef PADIM_H
#define PADIM_H

#include <torch/script.h>
#include <torch/torch.h>
#include "anomaly_map.h"
#include "multi_variate_gaussian.h"
#include "feature_extractor_engine.h"

using torch::indexing::Slice;
using torch::nn::Module;
using torch::nn::ModuleHolder;
namespace F = torch::nn::functional;

struct PadimModelImpl : Module
{
    PadimModelImpl(std::tuple<int, int> input_size):// , std::string backbone, std::vector<std::string> layers
    m_input_size(input_size)
    {
        // if (backbone == "efficientnet_v2_s")
        // {
        //     module = torch::jit::load("data/efficientnet_v2_s.pt");
        // }
        const std::string modelFile = "FeatureExtractor.engine";
        feature_extractor = std::shared_ptr<FeatureExtractor>(new FeatureExtractor(modelFile));
        auto [n_features_original, n_patches] = feature_extractor->_deduce_dim(input_size);
        auto idx = torch::arange(0, n_features_original, torch::kLong);
        register_buffer("idx", idx);

        anomaly_map_generator = AnomalyMapGenerator(input_size, 1.0);
        register_module("anomaly_map_generator", anomaly_map_generator);
        gaussian = MultiVariateGaussian(idx.size(0), n_patches);
        register_module("gaussian", gaussian);

        float a = std::numeric_limits<float>::max();
        min_val = torch::tensor({a});
        register_buffer("min_val", min_val);
        max_val = torch::tensor({-a});
        register_buffer("max_val", max_val);
        
    }

    torch::Tensor generate_embedding(std::vector<torch::Tensor> features)
    {
        torch::NoGradGuard no_grad;
        auto embeddings = features[0];
        int count = features.size();
        for (int i = 1; i < count; i++)
        {
            auto layer_embedding = features[i];
            int height = layer_embedding.size(2);
            int width = layer_embedding.size(3);
        
            layer_embedding = F::interpolate(layer_embedding, F::InterpolateFuncOptions().size(std::vector<int64_t>({height, width})).mode(torch::kNearest));
            embeddings = torch::cat({embeddings, layer_embedding}, 1);
        }
        auto idx = named_buffers()["idx"];
        embeddings = torch::index_select(embeddings, 1, idx);
        return embeddings;
    }

    torch::Tensor forward(cv::Mat frame)
    {
        auto dict = named_buffers();
        auto min_val = dict["min_val"];
        auto max_val = dict["max_val"];

        auto device = max_val.device();
        auto [inp_height, inp_width] = m_input_size;
        auto res = feature_extractor->prepareInput(frame, inp_width, inp_height);
        res = feature_extractor->infer();
        auto preds = feature_extractor->verifyOutput();
        std::vector<torch::Tensor> features;
        features.emplace_back(preds.to(device));
        auto embeddings = generate_embedding(features);

        

        auto max_tmp = torch::max(max_val, torch::max(embeddings));
        max_val.copy_(max_tmp);

        auto min_tmp = torch::min(min_val, torch::min(embeddings));
        min_val.copy_(min_tmp);

        return embeddings;
    }

    // int n_features_original, n_patches;
    std::shared_ptr<FeatureExtractor> feature_extractor;
    AnomalyMapGenerator anomaly_map_generator = nullptr;
    MultiVariateGaussian gaussian = nullptr;
    torch::Tensor max_val, min_val;
    std::tuple<int, int> m_input_size;
};
TORCH_MODULE(PadimModel);

#endif