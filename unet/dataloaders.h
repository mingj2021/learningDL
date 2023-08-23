#ifndef DATALOADERS_H
#define DATALOADERS_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <algorithm>
#include <filesystem>
#include <random>

namespace fs = std::filesystem;
using namespace torch::indexing;

using Data = std::vector<std::pair<std::string, std::string>>;

template <typename Data = at::Tensor, typename Target = at::Tensor>
struct ExampleCustom
{
    using DataType = Data;
    using TargetType = Target;

    ExampleCustom() = default;
    ExampleCustom(Data data, Target target)
        : data(std::move(data)), target(std::move(target))
    {
    }

    Data data;
    Target target;
};

class LoadImagesAndLabels : public torch::data::datasets::Dataset<LoadImagesAndLabels, ExampleCustom<>>
{
    Data data;

public:
    LoadImagesAndLabels(const Data &data) : data(data)
    {
    }

    ExampleCustom<> get(size_t index)
    {
        // std::cout << data[index].first << std::endl;
        cv::Mat img = cv::imread(data[index].first);
        int h0 = img.rows;
        int w0 = img.cols;
        cv::resize(img, img, cv::Size(int(w0 * 0.25), int(h0 * 0.25)), 0, 0, cv::INTER_LINEAR);
        // std::cout << img.size() << std::endl;
        cv::Mat mask = cv::imread(data[index].second, 0);
        cv::resize(mask, mask, cv::Size(int(w0 * 0.25), int(h0 * 0.25)), 0, 0, cv::INTER_LINEAR);

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255);

        torch::Tensor inputs =
            torch::from_blob(img.data, {img.rows, img.cols, img.channels()})
                .permute({2, 0, 1})
                .contiguous();

        // std::cout << "inputs = " << inputs.sizes() << std::endl;

        // cv::cvtColor(mask, mask, cv::COLOR_BGR2RGB);
        mask.convertTo(mask, CV_32F, 1.0 / 255);

        torch::Tensor targets =
            torch::from_blob(mask.data, {mask.rows, mask.cols})
                // .permute({2, 0, 1})
                .contiguous();

        return {inputs.clone(), targets.clone()};
    }

    torch::optional<size_t> size() const
    {
        return data.size();
    }

public:
};

Data readInfo(std::string input_dir = "/workspace/learningDL/unet/data/train_hq",
              std::string target_dir = "/workspace/learningDL/unet/data/clothes/labels/pixel_level_labels_colored")
{
    fs::path images_dir = input_dir;
    fs::path mask_dir = target_dir;
    Data train;

    for (const auto &entry : fs::directory_iterator(images_dir))
    {
        fs::path img = entry.path();
        fs::path mask = mask_dir / (img.stem().string() + ".png");
        if (fs::exists(mask))
        {
            train.emplace_back(std::make_pair(img, mask));
        }
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(train.begin(), train.end(), g);
    return train;
}

std::map<std::string, torch::Tensor> get_unique_colors(std::string input_f = "/workspace/learningDL/unet/data/clothes/class_dict.csv")
{
    std::map<std::string, torch::Tensor> name2rgb;
    std::ifstream file(input_f);
    std::string line;
    // remove 0 line(class_name,r,g,b)
    std::getline(file, line);
    while (std::getline(file, line))
    {
        std::stringstream linestream(line);
        std::string name, str_r, str_g, str_b;
        std::getline(linestream, name, ',');
        std::getline(linestream, str_r, ',');
        std::getline(linestream, str_g, ',');
        std::getline(linestream, str_b, ',');

        int r, g, b;
        std::istringstream(str_r) >> r;
        std::istringstream(str_g) >> g;
        std::istringstream(str_b) >> b;
        auto color = torch::tensor({r, g, b}).to(torch::kU8);
        name2rgb[name] = color;
    }
    return name2rgb;
}

template <typename T = ExampleCustom<>>
struct StackCustom;
template <>
struct StackCustom<ExampleCustom<>> : public torch::data::transforms::Collation<ExampleCustom<>>
{
    ExampleCustom<> apply_batch(std::vector<ExampleCustom<>> examples) override
    {
        std::vector<torch::Tensor> data, targets;
        data.reserve(examples.size());
        targets.reserve(examples.size());
        for (auto &example : examples)
        {
            // std::cout << example.data.sizes() << std::endl;
            auto t1 = example.data.unsqueeze(0);
            // std::cout << t1.sizes() << std::endl;
            data.push_back(std::move(t1));
            auto t2 = example.target.unsqueeze(0);
            // std::cout << t2.sizes() << std::endl;
            targets.push_back(std::move(t2));
        }
        // auto t1 = torch::cat(data);
        // auto t2 = torch::cat(targets);
        return {torch::cat(data), torch::cat(targets)};
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
int strhex2uint(std::string hex)
{
    int x;
    std::stringstream ss;
    ss << std::hex << hex;
    ss >> x;
    return x;
}
// mult class process
// download url: https://www.kaggle.com/datasets/balraj98/clothing-coparsing-dataset/download?datasetVersionNumber=1
class SegmentationDataSets : public torch::data::datasets::Dataset<SegmentationDataSets, ExampleCustom<>>
{
    Data data;
    std::map<std::string, torch::Tensor> name2rgb;

public:
    SegmentationDataSets(const Data &data, std::map<std::string, torch::Tensor> name2rgb) : data(data), name2rgb(name2rgb)
    {
    }

    ExampleCustom<> get(size_t index)
    {
        // std::cout << data[index].first << std::endl;
        cv::Mat img = cv::imread(data[index].first);
        // int h0 = 640;
        // int w0 = 640;
        // cv::resize(img, img, cv::Size(w0, h0), 0, 0, cv::INTER_LINEAR);
        // std::cout << img.size() << std::endl;

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255);

        torch::Tensor inputs =
            torch::from_blob(img.data, {img.rows, img.cols, img.channels()})
                .permute({2, 0, 1})
                .contiguous()
                .detach()
                .clone();

        cv::Mat img2 = cv::imread(data[index].second);
        // cv::resize(img2, img2, cv::Size(w0, h0), 0, 0, cv::INTER_LINEAR);

        cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
        torch::Tensor x = torch::from_blob(img2.data, {img2.rows, img2.cols, img2.channels()}, torch::kU8).detach().clone();
        int cls_num = 0;
        auto mask = torch::zeros({x.size(0), x.size(1)});
        for (const auto &[key, value] : name2rgb)
        {
            auto pos = x == value;
            pos = pos.all(2);
            mask.index_put_({pos}, cls_num++);
        }
        return {inputs, mask};
    }

    torch::optional<size_t> size() const
    {
        return data.size();
    }

public:
};
#endif