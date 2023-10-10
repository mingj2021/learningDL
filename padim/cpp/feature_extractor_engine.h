#ifndef FEATURE_EXTRACTOR_ENGINE_H
#define FEATURE_EXTRACTOR_ENGINE_H

#include "padim_utils.h"
#include <opencv2/opencv.hpp>
#include "commons/buffers.h"
#include "baseModel.h"

using torch::indexing::Slice;
using torch::nn::Module;
using torch::nn::ModuleHolder;
namespace F = torch::nn::functional;

class FeatureExtractor: public  BaseModel
{
public:
    FeatureExtractor(std::string modelFile);
    ~FeatureExtractor();

    std::tuple<int, int> _deduce_dim(std::tuple<int, int> input_size);
    int prepareInput(cv::Mat frame, int inp_width, int inp_height);
    bool infer();
    std::vector<at::Tensor> verifyOutput();

public:
};

FeatureExtractor::FeatureExtractor(std::string modelFile):BaseModel(modelFile)
{
    
}

FeatureExtractor::~FeatureExtractor()
{
}

std::tuple<int, int> FeatureExtractor::_deduce_dim(std::tuple<int, int> input_size)
{
    auto [height, width] = input_size;
    auto frame = cv::Mat::ones(cv::Size(width, height), CV_8UC3);
    std::cout << frame.size() << std::endl;
    prepareInput(frame, width, height);
    infer();
    auto preds = verifyOutput();
    int n_features_original = 0;
    int n_patches = preds[0].size(2) * preds[0].size(3);
    for (int i = 0; i < preds.size(); i++)
    {
        n_features_original += preds[i].size(1);
    }
    
    return std::make_tuple(n_features_original, n_patches);
}

int FeatureExtractor::prepareInput(cv::Mat frame, int inp_width, int inp_height)
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

    auto ret = mInOut["images"]->host2device((void *)(input_image_torch.data_ptr<float>()), true, stream);
    return ret;
}

bool FeatureExtractor::infer()
{   
    auto ret = context->enqueueV2(mDeviceBindings.data(), stream, nullptr);
    return ret;
}

std::vector<at::Tensor> FeatureExtractor::verifyOutput()
{
    std::vector<at::Tensor> outputs;
    for (auto &&name : mOutputsName)
    {
        auto dim0 = mEngine->getTensorShape(name.c_str());
        at::Tensor preds;
        preds = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
        mInOut[name.c_str()]->device2host((void *)(preds.data_ptr<float>()), true, stream);
        outputs.emplace_back(preds);
        break;
    }
    
    return outputs;
}

#endif