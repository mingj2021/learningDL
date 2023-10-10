#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include "yolo.h"
#include "loss.h"
#include <sstream>
#include "dataloaders.h"

using namespace torch::indexing;

template <typename DataLoader>
void train(ModuleHolder<DetectionModel> &network, DataLoader &loader, torch::optim::Optimizer &optimizer)
{
    auto device = network->parameters()[0].device();
    ComputeLoss compute_loss(network);
    network->train(true);

    for (auto &batch : loader)
    {
        auto inputs = batch.data.to(device);
        // std::cout << "inputs= " << inputs.sizes() << std::endl;
        auto targets = batch.target.to(device); //.reshape({-1, 6})
        // std::cout << "targets= " << targets.device() << std::endl;
        auto preds = network->forward(inputs);
        // std::cout << "preds= " << preds[0].device() << std::endl;
        auto vec_loss = compute_loss(preds, targets);
        // std::cout << "vec_loss= " << vec_loss.size() << std::endl;
        torch::Tensor loss = vec_loss[0];
        torch::Tensor loss2 = vec_loss[1];

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        std::cout << "loss = " << loss.item<float>() << " "
                  << "box_loss = " << loss2[0].item<float>() << " "
                  << "obj_loss= " << loss2[1].item<float>() << " "
                  << "cls_loss= " << loss2[2].item<float>() << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    YAML::Node config = YAML::LoadFile("/workspace/learningDL/yolov5/data/custom.yaml");
    std::map<std::string, float> hyp;
    hyp = config.as<std::map<std::string, float>>();

    // hyp['weight_decay'] *= batch_size * accumulate / nbs  // scale weight_decay
    // hyp['box'] *= 3 / nl  # scale to layers
    // hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
    // hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
    // hyp['label_smoothing'] = opt.

    float label_smoothing = 0.;
    int nbs = 64; // nominal batch size
    int batch_size = 4;
    int nc = 80;
    int nl = 3;
    int imgsz = 640;
    // torch::max(torch::round)
    auto accumulate = std::max(int(std::round(nbs / batch_size)), 1);
    hyp["weight_decay"] *= batch_size * accumulate / nbs;
    hyp["box"] *= 3 / nl;
    hyp["cls"] *= nc / 80 * 3 / nl;
    hyp["obj"] *= pow((imgsz / 640), 2) * 3 / nl;
    hyp["label_smoothing"] = label_smoothing;

    ModuleHolder<DetectionModel> model("/workspace/learningDL/yolov5/data/yolov5s.yaml", 3);
    model->hyp = hyp;
    model->load_weights("/workspace/learningDL/yolov5/data/yolov5s.weights");
    std::cout << model << std::endl;

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }
    torch::Device device(device_type);
    model->to(device);

    std::vector<torch::Tensor> g0, g1, g2;
    for (auto &layer : model->named_modules())
    {
        auto m_name = layer.key();
        auto m_v = layer.value();
        for (auto &it : m_v->named_parameters(false))
        {
            auto key = it.key();
            auto value = it.value();
            // std::cout << key << std::endl;
            if (key == "bias")
            {
                g2.push_back(value);
            }
            else if (key == "weight" && m_v->as<torch::nn::BatchNorm2dImpl>())
            {
                g1.push_back(value);
            }
            else
            {
                g0.push_back(value);
            }
        }
    }

    torch::optim::SGD optimizer(
        g2, torch::optim::SGDOptions(0.01).momentum(0.937).nesterov(true).weight_decay(1e-4));

    optimizer.add_param_group(torch::optim::OptimizerParamGroup(g0));

    optimizer.add_param_group(torch::optim::OptimizerParamGroup(g1));

    auto data = readInfo("/workspace/learningDL/yolov5/datasets/wafer/train.txt");
    auto train_set = LoadImagesAndLabels(data).map(StackCustom<>());
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_set), 4);
    int epoch = 100;
    for (size_t i = 0; i < epoch; i++)
    {
        std::cout << "epoch i = " << i << std::endl;
        train(model, *train_loader, optimizer);
        if (i % 5 == 0)
            torch::save(model, "yolov5s.pt");
            // torch::save(model, "yolov5_" + std::to_string(i) + ".pt");
        // test(model, data);
    }
    torch::save(model, "yolov5s.pt");
    return 0;
}
