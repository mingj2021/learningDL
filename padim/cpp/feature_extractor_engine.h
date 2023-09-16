#ifndef FEATURE_EXTRACTOR_ENGINE_H
#define FEATURE_EXTRACTOR_ENGINE_H

#include "padim_utils.h"
#include <opencv2/opencv.hpp>
#include "commons/buffers.h"

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
    void read_engine_file(std::string modelFile);
    int prepareInput(cv::Mat frame, int inp_width, int inp_height);
    bool infer();
    at::Tensor verifyOutput();

public:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    std::vector<void *> mDeviceBindings;
    std::map<std::string, std::unique_ptr<algorithms::DeviceBuffer>> mInOut;
};

FeatureExtractor::FeatureExtractor(std::string modelFile)
{
    read_engine_file(modelFile);
    context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        std::cerr << "create context error" << std::endl;
    }

    for (int i = 0; i < mEngine->getNbBindings(); i++)
    {
        auto dims = mEngine->getBindingDimensions(i);
        auto tensor_name = mEngine->getBindingName(i);
        std::cout << "tensor_name: " << tensor_name << std::endl;
        // dims2str(dims);
        nvinfer1::DataType type = mEngine->getBindingDataType(i);
        // index2srt(type);
        int vecDim = mEngine->getBindingVectorizedDim(i);
        // std::cout << "vecDim:" << vecDim << std::endl;
        if (-1 != vecDim) // i.e., 0 != lgScalarsPerVector
        {
            int scalarsPerVec = mEngine->getBindingComponentsPerElement(i);
            std::cout << "scalarsPerVec" << scalarsPerVec << std::endl;
        }
        auto vol = std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1}, std::multiplies<int64_t>{});
        std::unique_ptr<algorithms::DeviceBuffer> device_buffer{new algorithms::DeviceBuffer(vol, type)};
        mDeviceBindings.emplace_back(device_buffer->data());
        mInOut[tensor_name] = std::move(device_buffer);
    }
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
    int n_features_original = preds.size(1);
    int n_patches = preds.size(2) * preds.size(3);
    return std::make_tuple(n_features_original, n_patches);
}

void FeatureExtractor::read_engine_file(std::string modelFile)
{
    std::ifstream engineFile(modelFile.c_str(), std::ifstream::binary);
    assert(engineFile);

    int fsize;
    engineFile.seekg(0, engineFile.end);
    fsize = engineFile.tellg();
    engineFile.seekg(0, engineFile.beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    if (engineFile)
        std::cout << "all characters read successfully." << std::endl;
    else
        std::cout << "error: only " << engineFile.gcount() << " could be read" << std::endl;
    engineFile.close();

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr));
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

    auto ret = mInOut["images"]->host2device((void *)(input_image_torch.data_ptr<float>()), false);
    return ret;
}

bool FeatureExtractor::infer()
{   
    auto ret = context->executeV2(mDeviceBindings.data());
    return ret;
}

at::Tensor FeatureExtractor::verifyOutput()
{
    auto dim0 = mEngine->getTensorShape("output0");
    at::Tensor preds;
    preds = at::zeros({dim0.d[0], dim0.d[1], dim0.d[2], dim0.d[3]}, at::kFloat);
    mInOut["output0"]->device2host((void *)(preds.data_ptr<float>()), false);
    return preds;
}

#endif