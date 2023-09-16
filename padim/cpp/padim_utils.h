#ifndef PADIM_UTILS_H
#define PADIM_UTILS_H

#include <torch/torch.h>
namespace F = torch::nn::functional;
#include <iostream>
#include <fstream>

#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;
using namespace nvonnxparser;

#undef CHECK
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

void index2srt(nvinfer1::DataType dataType)
{
    switch (dataType)
    {
    case nvinfer1::DataType::kFLOAT:
        std::cout << "nvinfer1::DataType::kFLOAT" << std::endl;
        break;
    case nvinfer1::DataType::kHALF:
        std::cout << "nvinfer1::DataType::kHALF" << std::endl;
        break;
    case nvinfer1::DataType::kINT8:
        std::cout << "nvinfer1::DataType::kINT8" << std::endl;
        break;
    case nvinfer1::DataType::kINT32:
        std::cout << "nvinfer1::DataType::kINT32" << std::endl;
        break;
    case nvinfer1::DataType::kBOOL:
        std::cout << "nvinfer1::DataType::kBOOL" << std::endl;
        break;
    case nvinfer1::DataType::kUINT8:
        std::cout << "nvinfer1::DataType::kUINT8" << std::endl;
        break;

    default:
        break;
    }
}

void dims2str(nvinfer1::Dims dims)
{
    std::string o_s("[");
    for (size_t i = 0; i < dims.nbDims; i++)
    {
        if (i > 0)
            o_s += ", ";
        o_s += std::to_string(dims.d[i]);
    }
    o_s += "]";
    std::cout << o_s << std::endl;
}
class Logger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

void export_engine(std::string f)
{
    // create an instance of the builder
    std::unique_ptr<nvinfer1::IBuilder> builder(createInferBuilder(logger));
    // create a network definition
    // The kEXPLICIT_BATCH flag is required in order to import models using the ONNX parser.
    uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    // auto network = std::make_unique<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(flag));

    // Importing a Model Using the ONNX Parser
    // auto parser = std::make_unique<nvonnxparser::IParser>(createParser(*network, logger));
    std::unique_ptr<nvonnxparser::IParser> parser(createParser(*network, logger));

    // read the model file and process any errors
    parser->parseFromFile(f.c_str(),
                          static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    // create a build configuration specifying how TensorRT should optimize the model
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());

    // maximum workspace size
    // int workspace = 8;  // GB
    // config->setMaxWorkspaceSize(workspace * 1U << 30);
    config->setFlag(BuilderFlag::kGPU_FALLBACK);

    // config->setFlag(BuilderFlag::kFP16);

    // create an engine
    // auto serializedModel = std::make_unique<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    std::unique_ptr<nvinfer1::IHostMemory> serializedModel(builder->buildSerializedNetwork(*network, *config));
    std::cout << "serializedModel->size()" << serializedModel->size() << std::endl;
    std::ofstream outfile("FeatureExtractor.engine", std::ofstream::out | std::ofstream::binary);
    outfile.write((char *)serializedModel->data(), serializedModel->size());
}

torch::Tensor normalize_min_max(torch::Tensor targets, float threshold, float min_val, float max_val)
{
    // Apply min-max normalization and shift the values such that the threshold value is centered at 0.5.
    auto normalized = ((targets - threshold) / (max_val - min_val)) + 0.5;
    normalized = torch::minimum(normalized, torch::tensor({1}, normalized.options()));
    normalized = torch::maximum(normalized, torch::tensor({0}, normalized.options()));
    return normalized;
}

torch::Tensor standardize(torch::Tensor targets, float mean, float std, c10::optional<float> center_at = c10::nullopt)
{
    targets = torch::log(targets);
    auto standardized = (targets - mean) / std;
    if (center_at.has_value())
        standardized -= (center_at.value() - mean) / std;
    return standardized;
}

std::tuple<torch::Tensor, torch::Tensor> _normalize(torch::Tensor pred_scores, std::map<std::string, float> metadata, torch::Tensor anomaly_maps)
{
    // min max normalization
    if (metadata.find("min") != metadata.end() && metadata.find("max") != metadata.end())
    {
        anomaly_maps = normalize_min_max(
            anomaly_maps,
            metadata["pixel_threshold"],
            metadata["min"],
            metadata["max"]);
        pred_scores = normalize_min_max(
            pred_scores,
            metadata["image_threshold"],
            metadata["min"],
            metadata["max"]);
    }

    // standardize pixel scores

    // standardize image scores

    return std::make_tuple(anomaly_maps, pred_scores);
}

torch::Tensor connected_components(torch::Tensor image, int num_iterations = 100)
{
    int H = image.size(2);
    int W = image.size(3);

    auto mask = image == 1;
    int B = image.size(0);
    auto out = torch::arange(B * H * W, image.options()).view({-1, 1, H, W});
    // std::cout << out << std::endl;
    auto not_mask = torch::logical_not(mask);
    out.index_put_({not_mask}, 0);
    // std::cout << out << std::endl;

    for (size_t i = 0; i < num_iterations; i++)
    {
        auto t = F::max_pool2d(out, F::MaxPool2dFuncOptions(3).stride(1).padding(1));
        out.index_put_({mask}, t.index({mask}));
    }
    return out.view_as(image);
}

torch::Tensor connected_components_gpu(torch::Tensor image, int num_iterations = 1000)
{
    auto components = connected_components(image, num_iterations);

    auto [labels, labels_idx] = at::_unique(components);
    for (int new_label = 0; new_label < labels.size(0); new_label++)
    {
        int old_label = labels[new_label].item<int>();
        auto mask = components == old_label;
        components.index_put_({mask}, new_label);
    }
    return components.to(torch::kInt);
}
#endif