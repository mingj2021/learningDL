#ifndef DATALOADERS_H
#define DATALOADERS_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <algorithm>
#include "augmentations.h"
#include "general.h"

using namespace torch::indexing;

using Data = std::vector<std::pair<std::string, at::Tensor>>;

template <typename Data = at::Tensor, typename Target = at::Tensor>
struct ExampleCustom
{
    using DataType = Data;
    using TargetType = Target;

    ExampleCustom() = default;
    ExampleCustom(Data data, Target target, std::vector<std::string> path, torch::Tensor shape)
        : data(std::move(data)), target(std::move(target)), path(std::move(path)), shape(std::move(shape))
    {
    }

    Data data;
    Target target;
    std::vector<std::string> path;
    torch::Tensor shape;
};

class LoadImagesAndLabels : public torch::data::datasets::Dataset<LoadImagesAndLabels, ExampleCustom<>>
{
    Data data;

public:
    LoadImagesAndLabels(const Data &data, int img_size = 640, bool augment = true) : data(data), m_img_size(img_size), m_augment(augment)
    {
        m_mosaic = augment;
        m_mosaic_border.emplace_back(int(-1 * m_img_size / 2));
        m_mosaic_border.emplace_back(int(-1 * m_img_size / 2));
    }

    ExampleCustom<> get(size_t index)
    {
        cv::Mat img;
        torch::Tensor labels;
        std::vector<float> shapes;
        if (m_mosaic)
        {
            auto rets = load_mosaic(index);
            img = std::get<0>(rets);
            labels = std::get<1>(rets);
        }
        else
        {
            std::string path = data[index].first;
            labels = data[index].second.detach().clone();
            auto [mat, sz0, sz1] = load_image(index);
            auto pad_info = letterbox2(mat, img, cv::Size(640, 640), cv::Scalar(114, 114, 114), false, false, false);
            const float ratio_w = pad_info[0];
            const float ratio_h = pad_info[1];

            const float pad_w = pad_info[2];
            const float pad_h = pad_info[3];
            shapes = {float(sz0.height), float(sz0.width),
                                  float(sz1.height) / float(sz0.height), float(sz1.width) / float(sz0.width),
                                  pad_w, pad_h};
            
            labels.index({Slice(), Slice(2)}) = xywhn2xyxy(labels.index({Slice(), Slice(2)}), sz1.width * ratio_w, sz1.height * ratio_h, pad_w, pad_h);
        }
        if (labels.size(0))
        {
            labels.index({Slice(), Slice(2)}) = xyxy2xywhn(labels.index({Slice(), Slice(2)}), img.cols, img.rows, true, 1e-3);
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_32F, 1.0 / 255);

        torch::Tensor inputs =
            torch::from_blob(img.data, {img.rows, img.cols, img.channels()})
                .permute({2, 0, 1})
                .contiguous();
        // std::cout << labels << std::endl;
        std::vector<std::string> path;
        path.emplace_back(data[index].first);
        return {inputs.clone(), labels, path, torch::tensor(shapes)};
    }

    torch::optional<size_t> size() const
    {
        return data.size();
    }

    std::tuple<cv::Mat, cv::Size, cv::Size> load_image(size_t index)
    {
        // std::string paths_, int img_size = 640
        std::string paths_ = data[index].first;
        auto mat = cv::imread(paths_);
        int h0 = mat.rows;
        int w0 = mat.cols;
        float r = 1.0 * m_img_size / std::max(h0, w0);
        // std::cout << "r " << r << std::endl;
        // float t = 1.0;
        // std::cout << "t " << (t == 1.0) << std::endl;
        if (r != 1.0)
        {
            int interp;
            if (r > 1)
                interp = cv::INTER_LINEAR;
            else
                interp = cv::INTER_AREA;
            cv::resize(mat, mat, cv::Size(int(w0 * r), int(h0 * r)), 0, 0, interp);
            // std::cout << "interp " << interp << std::endl;
            // std::cout << "mat " << mat.size() << std::endl;
        }
        // std::cout << "mat " << mat.size() << std::endl;
        return std::make_tuple(mat, cv::Size(w0, h0), cv::Size(mat.cols, mat.rows));
    }

    std::tuple<cv::Mat, torch::Tensor> load_mosaic(size_t index)
    {
        std::vector<torch::Tensor> vec_labels4;
        int s = m_img_size;
        int yc = torch::zeros({1}).uniform_(-m_mosaic_border[0], 2 * s + m_mosaic_border[0]).item<int>();
        int xc = torch::zeros({1}).uniform_(-m_mosaic_border[1], 2 * s + m_mosaic_border[1]).item<int>();
        auto indices = torch::randperm(data.size()).index({Slice(None, 4)});
        indices[0] = int(index);
        auto img4 = cv::Mat(s * 2, s * 2, CV_8UC3, cv::Scalar(114, 114, 114));
        for (size_t i = 0; i < int(indices.numel()); i++)
        {
            /* code */
            index = indices[i].item<int>();
            auto [mat, sz0, sz1] = load_image(index);
            int h, w;
            h = sz1.height;
            w = sz1.width;
            int x1a, y1a, x2a, y2a;
            int x1b, y1b, x2b, y2b;
            if (i == 0)
            {
                x1a = std::max(xc - w, 0);
                y1a = std::max(yc - h, 0);
                x2a = xc;
                y2a = yc;

                x1b = w - (x2a - x1a);
                y1b = h - (y2a - y1a);
                x2b = w;
                y2b = h;
            }
            else if (i == 1)
            {
                x1a = xc;
                y1a = std::max(yc - h, 0);
                x2a = std::min(xc + w, s * 2);
                y2a = yc;

                x1b = 0;
                y1b = h - (y2a - y1a);
                x2b = std::min(w, x2a - x1a);
                y2b = h;
            }
            else if (i == 2)
            {
                x1a = std::max(xc - w, 0);
                y1a = yc;
                x2a = xc;
                y2a = std::min(s * 2, yc + h);

                x1b = w - (x2a - x1a);
                y1b = 0;
                x2b = w;
                y2b = std::min(y2a - y1a, h);
            }
            else if (i == 3)
            {
                x1a = xc;
                y1a = yc;
                x2a = std::min(xc + w, s * 2);
                y2a = std::min(s * 2, yc + h);

                x1b = 0;
                y1b = 0;
                x2b = std::min(w, x2a - x1a);
                y2b = std::min(y2a - y1a, h);
            }

            auto src_ = mat(cv::Range(y1b, y2b), cv::Range(x1b, x2b));
            auto dst_ = img4(cv::Range(y1a, y2a), cv::Range(x1a, x2a));
            src_.copyTo(dst_);
            int padw = x1a - x1b;
            int padh = y1a - y1b;

            torch::Tensor labels = data[index].second.detach().clone();
            labels.index({Slice(), Slice(2)}) = xywhn2xyxy(labels.index({Slice(), Slice(2)}), w, h, padw, padh);
            vec_labels4.emplace_back(labels);
            // cv::namedWindow("img4", 0);
            // cv::imshow("img4", img4);
            // cv::waitKey();
        }
        auto labels4 = torch::cat(vec_labels4, 0).clamp_(0, 2 * s);
        // std::cout << "yc " << yc << std::endl;
        // std::cout << "xc " << xc << std::endl;
        // std::cout << indices << std::endl;
        // std::cout << labels4 << std::endl;
        auto [img4_, labels4_] = random_perspective(img4, labels4, torch::zeros({1}), 0.0, 0.1, 0.5, 0.0, 0.0, m_mosaic_border);
        // for (int i = 0; i < labels4_.size(0); i++)
        // {
        //     auto x1 = labels4_[i][2].item().toFloat();
        //     auto y1 = labels4_[i][3].item().toFloat();
        //     auto x2 = labels4_[i][4].item().toFloat();
        //     auto y2 = labels4_[i][5].item().toFloat();
        //     // auto x1 = cx - w / 2;
        //     // auto y1 = cy - h / 2;
        //     // auto x2 = cx + w / 2;
        //     // auto y2 = cy + h / 2;
        //     // std::cout << "score: " << score << "cls " << cls << std::endl;
        //     cv::Point p1(x1, y1);
        //     cv::Point p2(x2, y2);
        //     cv::rectangle(img4_, p1, p2,
        //                   cv::Scalar(255, 0, 0));
        // }
        // cv::namedWindow("src", 0);
        // cv::imshow("src", img4_);
        // cv::waitKey(0);

        return std::make_tuple(img4_, labels4_);
    }

public:
    int m_img_size;
    bool m_augment;
    bool m_mosaic;
    std::vector<int> m_mosaic_border;
};

Data readInfo(std::string fs = "C:/Users/77274/projects/datasets/coco128/train.txt")
{
    Data train;

    std::ifstream in(fs);
    assert(in.is_open());

    std::string path, path1;

    while (true)
    {
        in >> path1;
        path = path1;
        std::string s_label;
        std::string str1("images");
        std::string str2("labels");
        auto found = path.find(str1);
        if (found != std::string::npos)
        {
            s_label = path.replace(found, str1.size(), str2);
        }
        str1 = ".jpg";
        str2 = ".txt";
        found = path.find(str1);
        if (found != std::string::npos)
        {
            s_label = path.replace(found, str1.size(), str2);
        }
        std::ifstream stream(s_label);
        std::string line;
        std::vector<at::Tensor> vec_labels;

        while (std::getline(stream, line))
        {
            std::istringstream iss(line);
            float cls = 0;
            float id = 0.0;
            float x = 0.0, y = 0.0, w = 0.0, h = 0.0;
            iss >> cls >> x >> y >> w >> h;
            // std::cout << cls << " " << x << " " << y << " " << w << " " << h << " " << std::endl;
            at::Tensor t_label = torch::tensor({{id, cls, x, y, w, h}});
            vec_labels.emplace_back(t_label);
        }
        int nl = vec_labels.size();
        at::Tensor labels = at::zeros({3, 6});
        if (nl > 0)
        {
            labels = at::cat(vec_labels, 0);
            train.push_back(std::make_pair(path1, labels));
        }
        if (in.eof())
            break;
    }

    // std::shuffle(train.begin(), train.end());
    return train;
}

template <typename T = ExampleCustom<>>
struct StackCustom;
template <>
struct StackCustom<ExampleCustom<>> : public torch::data::transforms::Collation<ExampleCustom<>>
{
    ExampleCustom<> apply_batch(std::vector<ExampleCustom<>> examples) override
    {
        std::vector<torch::Tensor> data, targets, shape;
        std::vector<std::string> path;
        data.reserve(examples.size());
        targets.reserve(examples.size());
        int i = 0;
        for (auto &example : examples)
        {
            data.push_back(std::move(example.data));
            example.target.index({Slice(), 0}) = float(i++);
            targets.push_back(std::move(example.target));
            path.insert(path.end(), example.path.begin(), example.path.end());
            shape.push_back(std::move(example.shape));
        }
        return {torch::stack(data), torch::cat(targets), path, torch::stack(shape)};
    }
};
#endif