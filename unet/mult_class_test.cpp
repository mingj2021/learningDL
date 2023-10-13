#include "unet_model.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include "dataloaders.h"
#include <vector>

using torch::indexing::Slice;

void mask2image(torch::Tensor mask, std::map<std::string, torch::Tensor> name2rgb, std::string output)
{
    torch::Tensor out;
    // rgb format
    if (name2rgb.begin()->second.size(0) == 3)
    {
        out = torch::zeros({mask.size(0), mask.size(1), 3}, torch::kU8).to(mask.device());
    }
    // color = [0,1]
    else
    {
        out = torch::zeros({mask.size(0), mask.size(1)}, torch::kU8).to(mask.device());
    }
    int cls_num = 0;

    for (const auto &[key, value] : name2rgb)
    {
        auto idx = mask == cls_num++;
        out.index_put_({idx}, value.to(mask.device()).to(torch::kU8));
    }
    out = out.clamp(0, 255).to(torch::kU8).to(torch::kCPU);
    cv::Mat img_(out.size(0), out.size(1), CV_8UC3, out.data_ptr<uchar>());
    cv::cvtColor(img_, img_, cv::COLOR_RGB2BGR);
    cv::imwrite(output, img_);
}

template <typename DataLoader>
void test(ModuleHolder<UNet> &network, DataLoader &loader, std::map<std::string, torch::Tensor> name2rgb)
{
    auto device = network->parameters()[0].device();
    network->eval();
    int i = 0;
    for (auto &batch : loader)
    {
        auto inputs = batch.data.to(device).to(torch::kHalf); //
        // std::cout << "inputs= " << inputs.sizes() << std::endl;
        auto targets = batch.target.to(device).to(torch::kLong); //
        // auto targets = torch::empty({inputs.size(0),inputs.size(2),inputs.size(3)},torch::kLong).random_(9).to(device);
        // std::cout << "targets= " << targets.sizes() << std::endl;
        auto preds = network->forward(inputs).to(torch::kFloat);
        // auto preds = torch::randn({inputs.size(0),9,inputs.size(2),inputs.size(3)},at::TensorOptions().requires_grad(true)).to(device);
        // std::cout << "preds= " << preds.sizes() << std::endl;
        auto mask = preds.argmax(1).squeeze(0);
        // auto [b1,b2,b3] = torch::unique_dim(mask,1);
        // std::cout << mask.sizes() << std::endl;
        // torch::Tensor unique_colors;
        // torch::load(unique_colors, "colors.pth");
        std::string output = "t_" + std::to_string(i++) + ".png";
        // mask2image(mask, name2rgb, output);
        torch::Tensor mask_ = mask.clamp(0, 255).to(torch::kU8).to(torch::kCPU)*255;
        cv::Mat img_(mask_.size(0), mask_.size(1), CV_8UC1, mask_.data_ptr<uchar>());
        cv::imwrite(output, img_);
        std::cout << output << std::endl;
        // break;
    }
}

int main(int argc, char const *argv[])
{
    // {
    //     auto d = torch::zeros({5,5});
    //     torch::save(d,"self.pth");
    //     torch::Tensor c;
    //     torch::load(c,"self.pth");
    //     std::cout << c << std::endl;
    // return 0;
    // }
    // {
    //     auto name2rgb = get_unique_colors();
    //     // for (const auto &[key, value] : name2rgb)
    //     //     std::cout << '[' << key << "] = " << value << "; ";
    //     auto data = readInfo("/workspace/learningDL/unet/data/clothes/images",
    //                          "/workspace/learningDL/unet/data/clothes/labels/pixel_level_labels_colored");
    //     auto train_set = SegmentationDataSets(data, name2rgb);
    //     std::cout << "----------------------0" << std::endl;
    //     auto d = train_set.get(0);
    //     auto mask = d.target;
    //     mask2image(mask,name2rgb,"test.png");
    //     return 0;
    // }
    // {
    //     fs::path images_dir = "/workspace/learningDL/unet/data/Legs Segmentation/masks";
    //     std::vector<torch::Tensor> vec_values;
    //     for (const auto &entry : fs::directory_iterator(images_dir))
    //     {
    //         fs::path img = entry.path();
    //         if (fs::exists(img))
    //         {
    //             auto img2 = cv::imread(img);
    //             cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    //             torch::Tensor x = torch::from_blob(img2.data, {img2.rows, img2.cols, img2.channels()}, torch::kU8).detach().clone();
    //             x = x.reshape({-1,3});
    //             auto [b1,b2,b3] = at::unique_dim(x,0);
    //             vec_values.emplace_back(b1.unsqueeze(0));
    //         }
    //     }
    //     std::cout << "---------" << std::endl;
    //     auto values = torch::cat(vec_values,1).squeeze(0);
    //     auto [b1,b2,b3] = at::unique_dim(values,0);
    //     auto img2 = cv::imread("/workspace/learningDL/unet/data/train_masks/0cdf5b5d0ce1_02_mask.jpg");
    //     cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
    //     torch::Tensor x = torch::from_blob(img2.data, {img2.rows, img2.cols, img2.channels()}, torch::kU8).detach().clone();
    //     auto pos = x == b1[0];
    //     std::cout << pos.sizes() << std::endl;
    //     pos = pos.all(2);
    //     std::cout << pos.sizes() << std::endl;

    //     // std::cout << b1 << std::endl;
    //     // auto idx = b1 == torch::tensor({0,0,0});
    //     // std::cout << "--------" << std::endl;
    //     // std::cout << idx.all(1) << std::endl;

    //     return 0;
    // }
    auto t0 = std::time(nullptr);
    auto tm = *std::localtime(&t0);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y_%m_%d_%S_%M_%H");
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

    ModuleHolder<UNet> model(3, 2, false);
    torch::load(model, "2023_10_13_02_29_13_unet.pth");
    model->to(device);
    model->to(torch::kHalf);
    // auto data = readInfo("/workspace/learningDL/unet/data/clothes/images",
    //                         "/workspace/learningDL/unet/data/clothes/labels/pixel_level_labels_colored");
    // auto name2rgb = get_unique_colors("/workspace/learningDL/unet/data/clothes/class_dict.csv");
    // auto test_set = SegmentationDataSets(data, name2rgb).map(StackCustom<>());
    auto data = readInfo("/workspace/learningDL/unet/data/wafer/1/images",
                        "/workspace/learningDL/unet/data/wafer/1/labels");
    auto name2rgb = get_unique_colors("/workspace/learningDL/unet/data/clothes/class_dict.csv");
    auto test_set = LoadImagesAndLabels(data).map(StackCustom<>());
    auto test_loader =
        torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(test_set), 1);
    test(model, *test_loader, name2rgb);

    // auto img = cv::imread("/workspace/learningDL/unet/data/Legs Segmentation/img/1.png");
    // int h0 = 640;
    // int w0 = 640;
    // cv::resize(img, img, cv::Size(w0, h0), 0, 0, cv::INTER_LINEAR);
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // img.convertTo(img, CV_32F, 1.0 / 255);

    // torch::Tensor inputs =
    //     torch::from_blob(img.data, {img.rows, img.cols, img.channels()})
    //         .permute({2, 0, 1})
    //         .unsqueeze(0)
    //         .contiguous()
    //         .to(device)
    //         .to(torch::kHalf);
    // auto rets = model->forward(inputs).to(torch::kFloat);
    // auto mask = rets.argmax(1).squeeze(0);
    // torch::Tensor unique_colors;
    // torch::load(unique_colors, "colors.pth");
    // mask2image(mask, unique_colors);

    // for (int i = 0; i < 8; i++)
    // {
    //     auto pred = rets.index({Slice(), i, Slice(), Slice()});
    //     auto h = pred.size(1);
    //     auto w = pred.size(2);
    //     pred = pred.reshape({h, w}).contiguous() * 255;
    //     pred = pred.clamp(0, 255).to(torch::kU8).to(torch::kCPU);
    //     cv::Mat img_(pred.size(0), pred.size(1), CV_8U, pred.data_ptr<uchar>());
    //     std::cout << img_.size() << std::endl;
    //     cv::imwrite("out" + std::to_string(i) + ".png", img_);
    // }

    // cv::imshow("1", img_);
    // cv::waitKey();
}