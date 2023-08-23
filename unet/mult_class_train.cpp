#include "unet_model.h"
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include "dataloaders.h"
#include "loss.h"
#include <chrono>

using torch::Tensor;
using torch::nn::CrossEntropyLoss;
using torch::nn::BCEWithLogitsLoss;
using torch::nn::NLLLoss;

template <typename DataLoader>
void train(ModuleHolder<UNet> &network, DataLoader &loader, torch::optim::Optimizer &optimizer)
{
    auto device = network->parameters()[0].device();
    // ComputeLoss compute_loss(network);
    AnyModule criterion;
    // F::binary_cross_entropy
    if (network->m_n_classes > 1)
    {
        criterion = AnyModule(CrossEntropyLoss());
        // criterion = AnyModule(NLLLoss());
    }
    else
    {
        criterion = AnyModule(BCEWithLogitsLoss());
    }
    network->train(true);
    for (auto &batch : loader)
    {
        auto inputs = batch.data.to(device).to(torch::kHalf);//
        // std::cout << "inputs= " << inputs.sizes() << std::endl;
        auto targets = batch.target.to(device).to(torch::kLong); //
        // std::cout << "targets= " << targets.sizes() << std::endl;
        auto preds = network->forward(inputs).to(torch::kFloat);
        // std::cout << "preds= " << preds.sizes() << std::endl;
        
        Tensor loss;
        if (network->m_n_classes == 1)
        {
            std::cout << "----------------------------" << std::endl;
            loss = criterion.forward(preds.squeeze(1), targets);
            // loss += dice_loss(preds.squeeze(1), targets,false);
        }
        else
        {
            // auto input = torch::log_softmax(preds,1);
            loss = criterion.forward(preds, targets);
            // st::cout << "loss " << loss << std::endl;
            // loss += DiceLoss(preds, targets,network->m_n_classes);
            loss += dice_loss(F::softmax(preds,F::SoftmaxFuncOptions(1)).to(torch::kFloat),F::one_hot(targets,network->m_n_classes).permute({0,3,1,2}).to(torch::kFloat),true);
        }
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        std::cout << '\r' << "loss= " << loss.cpu().item<float>() << std::flush;
    }
}

int main(int argc, char const *argv[])
{
    auto t0 = std::time(nullptr);
    auto tm = *std::localtime(&t0);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y_%m_%d_%H_%M_%S");
    auto start_t = oss.str();
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

    ModuleHolder<UNet> model(3, 59, false);
    model->to(device);
    model->to(torch::kHalf);
    std::cout << model << std::endl;
    torch::optim::SGD optimizer(
        model->parameters(), torch::optim::SGDOptions(1e-3).momentum(0.937).nesterov(true).weight_decay(1e-4));//
    auto data = readInfo("/workspace/learningDL/unet/data/clothes/images",
                            "/workspace/learningDL/unet/data/clothes/labels/pixel_level_labels_colored");
    auto name2rgb = get_unique_colors();
    auto train_set = SegmentationDataSets(data, name2rgb).map(StackCustom<>());
    auto train_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(train_set), 1);
    for (size_t i = 0; i < 600; i++)
    {
        std::cout << "epoch = "  << i << std::endl;
        train(model, *train_loader, optimizer);
        torch::save(model, start_t + "_unet.pth");
    }
    std::cout << "done" << std::endl;
    return 0;
}
