#ifndef FEATURE_EXTRACTOR_TORCHSCRIPT_H
#define FEATURE_EXTRACTOR_TORCHSCRIPT_H
#include <opencv2/opencv.hpp>
#include <torch/script.h>

using torch::indexing::Slice;
using torch::nn::Module;
using torch::nn::ModuleHolder;
namespace F = torch::nn::functional;

class FeatureExtractor
{
public:
    FeatureExtractor(std::string modelFile);
    ~FeatureExtractor();

    std::tuple<int, int> _deduce_dim(std::tuple<int, int> input_size);
    torch::Tensor prepareInput(cv::Mat frame, int inp_width, int inp_height);
    std::vector<at::Tensor> forward(torch::Tensor);

public:
    torch::jit::script::Module model;
};

FeatureExtractor::FeatureExtractor(std::string modelFile)
{
    model = torch::jit::load(modelFile);
}

FeatureExtractor::~FeatureExtractor()
{
}

std::tuple<int, int> FeatureExtractor::_deduce_dim(std::tuple<int, int> input_size)
{
    auto [height, width] = input_size;
    auto frame = cv::Mat::ones(cv::Size(width, height), CV_8UC3);
    auto inputs = prepareInput(frame, width, height);
    auto preds = forward(inputs);
    int n_features_original = 0;
    int n_patches = preds[0].size(2) * preds[0].size(3);
    for (int i = 0; i < preds.size(); i++)
    {
        n_features_original += preds[i].size(1);
    }
    
    return std::make_tuple(n_features_original, n_patches);
}

torch::Tensor FeatureExtractor::prepareInput(cv::Mat frame, int inp_width, int inp_height)
{
    cv::Mat im_sz;

    cv::resize(frame, im_sz, cv::Size(inp_width, inp_height));
    cv::cvtColor(im_sz, im_sz, cv::COLOR_BGR2RGB);
    im_sz.convertTo(im_sz, CV_32F, 1.0 / 255);
    at::Tensor input_image_torch =
        at::from_blob(im_sz.data, {im_sz.rows, im_sz.cols, im_sz.channels()})
            .permute({2, 0, 1})
            .contiguous()
            .unsqueeze(0);
    return input_image_torch;
}

std::vector<at::Tensor> FeatureExtractor::forward(torch::Tensor x)
{
    c10::List<at::Tensor> preds = model.forward({x}).toTensorList();
    std::vector<at::Tensor> outputs;
    for (auto &&i : preds)
    {
        outputs.emplace_back(i);
    }
    return outputs;
}
#endif