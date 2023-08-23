#ifndef AUGMENTATIONS_H
#define AUGMENTATIONS_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <cmath>

using namespace torch::indexing;

std::vector<float> letterbox(const cv::Mat &src, cv::Mat &dst, const cv::Size &out_size)
{
    auto in_h = static_cast<float>(src.rows);
    auto in_w = static_cast<float>(src.cols);
    float out_h = out_size.height;
    float out_w = out_size.width;

    float scale = std::min(out_w / in_w, out_h / in_h);

    int mid_h = static_cast<int>(in_h * scale);
    int mid_w = static_cast<int>(in_w * scale);

    cv::resize(src, dst, cv::Size(mid_w, mid_h));

    int top = (static_cast<int>(out_h) - mid_h) / 2;
    int down = (static_cast<int>(out_h) - mid_h + 1) / 2;
    int left = (static_cast<int>(out_w) - mid_w) / 2;
    int right = (static_cast<int>(out_w) - mid_w + 1) / 2;

    cv::copyMakeBorder(dst, dst, top, down, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    std::vector<float> pad_info{static_cast<float>(left), static_cast<float>(top), scale};
    return pad_info;
}

std::vector<float> letterbox2(const cv::Mat &src, cv::Mat &dst, const cv::Size &new_size = cv::Size(640, 640),
                              const cv::Scalar color = cv::Scalar(114, 114, 114), bool align = true, bool scaleFill = false, bool scaleup = true, int stride = 32)
{
    // Resize and pad image while meeting stride-multiple constraints
    cv::Size c_size(src.cols, src.rows); // current shape [width, height]

    // Scale ratio (new / old)
    float r = std::min(1.0 * new_size.height / c_size.height, 1.0 * new_size.width / c_size.width);
    if (!scaleup) // only scale down, do not scale up (for better val mAP)
        r = std::min(r * 1.0, 1.0);

    float ratio_w = r;
    float ratio_h = r;

    float new_unpad_w = std::round(c_size.width * r);
    float new_unpad_h = std::round(c_size.height * r);
    float dw, dh; //  wh padding
    dw = new_size.width - new_unpad_w;
    dh = new_size.height - new_unpad_h;

    if (align) //  minimum rectangle
    {
        dw = int(dw) % stride;
        dh = int(dh) % stride;
    }
    else if (scaleFill)
    {
        dw = dh = 0;
        new_unpad_w = new_size.width;
        new_unpad_h = new_size.height;
        ratio_w = new_size.width / c_size.width;
        ratio_h = new_size.height / c_size.height;
    }

    dw /= 2.0;
    dh /= 2.0;
    cv::Size new_unpad(new_unpad_w, new_unpad_h);
    if (c_size != new_unpad)
    {
        cv::resize(src, dst, new_unpad, 0, 0, cv::INTER_LINEAR);
    }
    else
    {
        dst = src;
    }

    int top = int(std::round(dh - 0.1));
    int bottom = int(std::round(dh + 0.1));
    int left = int(std::round(dw - 0.1));
    int right = int(std::round(dw + 0.1));
    cv::copyMakeBorder(dst, dst, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    std::vector<float> pad_info{static_cast<float>(ratio_w), static_cast<float>(ratio_h), static_cast<float>(dw), static_cast<float>(dh)};
    return pad_info;
}

torch::Tensor box_candidates(torch::Tensor box1,torch::Tensor box2, float wh_thr=2,float ar_thr=100,float area_thr=0.1,float eps=1e-16)
{
    auto w1 = box1[2] - box1[0];
    auto h1 = box1[3] - box1[1];
    auto w2 = box2[2] - box2[0];
    auto h2 = box2[3] - box2[1];
    auto ar = torch::max(w2 / (h2 + eps), h2 / (w2 + eps));
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr);
}

std::tuple<cv::Mat, torch::Tensor> random_perspective(cv::Mat im, torch::Tensor targets, torch::Tensor segments,
                                                      float degrees, float translate, float scale, float shear, float perspective, std::vector<int> border)
{
    // std::cout << targets << std::endl;
    int height = im.rows + border[0] * 2;
    int width = im.cols + border[1] * 2;

    auto C = torch::eye(3);
    C.index({0, 2}) = -im.cols / 2;
    C.index({1, 2}) = -im.rows / 2;
    // std::cout << C << std::endl;

    auto P = torch::eye(3);
    P.index({2, 0}) = torch::zeros({1}).uniform_(-perspective, perspective).item<float>();
    P.index({2, 1}) = torch::zeros({1}).uniform_(-perspective, perspective).item<float>();
    // std::cout << P << std::endl;

    // auto R = torch::eye(3);
    cv::Mat cv_R = cv::Mat::eye(3, 3, CV_64F);
    // std::cout << R << std::endl;
    auto a = torch::zeros({1}).uniform_(-degrees, degrees).item<float>();
    auto s = torch::zeros({1}).uniform_(1 - scale, 1 + scale).item<float>();
    auto rot = cv::getRotationMatrix2D(cv::Point2f(0, 0), a, s);
    // std::cout << rot << std::endl;
    auto R_ = cv_R(cv::Range(0, 2), cv::Range::all());
    rot.copyTo(R_);
    auto R = torch::from_blob(cv_R.data, {cv_R.rows, cv_R.cols}, torch::kFloat64).to(torch::kFloat32); //, cv_R.channels()
    // std::cout << cv_R << std::endl;
    // std::cout << R << std::endl;
    // R_ = rot;
    // std::cout << R_ << std::endl;
    // std::cout << R << std::endl;
    auto S = torch::eye(3);
    float ntmp = torch::zeros({1}).uniform_(-shear, shear).item<float>();
    ntmp = tan(ntmp * M_PI / 180);
    S.index({0, 1}) = ntmp;
    ntmp = torch::zeros({1}).uniform_(-shear, shear).item<float>();
    ntmp = tan(ntmp * M_PI / 180);
    S.index({1, 0}) = ntmp;

    auto T = torch::eye(3);
    T.index({0, 2}) = torch::zeros({1}).uniform_(0.5 - translate, 0.5 + translate).item<float>() * width;
    T.index({1, 2}) = torch::zeros({1}).uniform_(0.5 - translate, 0.5 + translate).item<float>() * height;

    // T @ S @ R @ P @ C;
    auto M = T.matmul(S).matmul(R).matmul(P).matmul(C);
    M = M.to(torch::kFloat64);
    cv::Mat cv_M(M.size(0), M.size(1), CV_64F, M.data_ptr<double>());
    // std::cout << "-------------" << std::endl;
    // std::cout << M << std::endl;
    // std::cout << "-------------" << std::endl;
    // std::cout << cv_M << std::endl;
    // auto b = (M != torch::eye(3));
    // std::cout << "-------------" << std::endl;
    // std::cout << b << std::endl;
    // std::cout << torch::any(b).item<bool>() << std::endl;
    // std::cout << (!torch::equal(M, torch::eye(3))) << std::endl;
    if (border[0] != 0 || border[1] != 0 || (!torch::equal(M, torch::eye(3))))
    {
        if (perspective)
        {
            cv::warpPerspective(im, im, cv_M, cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }
        else
        {
            cv::warpAffine(im, im, cv_M(cv::Range(0, 2), cv::Range::all()), cv::Size(width, height), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
        }
    }
    // cv::namedWindow("im", 0);
    // cv::imshow("im", im);
    // cv::waitKey();
    // std::cout << im.size() << std::endl;
    int n = targets.size(0);
    if (n)
    {
        bool use_segments = false;
        auto t_new = torch::zeros({n, 4});
        if (use_segments)
        {
        }
        else
        {
            auto xy = torch::ones({n * 4, 3});
            // std::cout << xy << std::endl;
            xy.index({Slice(),Slice(None,2)}) = targets.index({Slice(), torch::tensor({1+1, 2+1, 3+1, 4+1, 1+1, 4+1, 3+1, 2+1})}).reshape({n * 4, 2});
            // std::cout << xy << std::endl;
            xy = xy.matmul(M.to(torch::kFloat32).t());
            // std::cout << xy << std::endl;
            if(perspective)
            {
                xy = xy.index({Slice(),Slice(None,2)}) / xy.index({Slice(),Slice(2,3)});
                xy = xy.reshape({n, 8});
            }
            else
            {
                xy = xy.index({Slice(),Slice(None,2)});
                xy = xy.reshape({n, 8});
            }
            // std::cout << "xy-------------------------" << std::endl;
            auto x = xy.index({Slice(), torch::tensor({0, 2, 4, 6})});
            auto y = xy.index({Slice(), torch::tensor({1, 3, 5, 7})});
            auto [xminV,xminInd] = x.min(1);
            auto [yminV,yminInd] = y.min(1);
            auto [xmaxV,xmaxInd] = x.max(1);
            auto [ymaxV,ymaxInd] = y.max(1);

            // std::cout << xminV << std::endl;
            // std::cout << "xminV" << std::endl;
            t_new = torch::cat({xminV,yminV,xmaxV,ymaxV}).reshape({4,n}).t();
            t_new.index_put_({Slice(), torch::tensor({0, 2})},t_new.index({Slice(), torch::tensor({0, 2})}).clamp_(0,width));
            t_new.index_put_({Slice(), torch::tensor({1, 3})}, t_new.index({Slice(), torch::tensor({1, 3})}).clamp_(0,height));
        }
        // std::cout << s << std::endl;
        // std::cout << t_new << std::endl;
        auto box1 = targets.index({Slice(),Slice(1+1,5+1)}).t() * s;
        auto box2 = t_new.t();
        float area_thr;
        if(use_segments)
        {
            area_thr=0.01;
        }
        else
        {
            area_thr=0.10;
        }
        auto i = box_candidates(box1,box2,2,100,area_thr);
        // std::cout << targets << std::endl;
        // std::cout << i << std::endl;
        targets = targets.index({i});
        targets.index({Slice(),Slice(1+1,5+1)}) = t_new.index({i});
        // std::cout << targets << std::endl;
        // std::cout << t_new.index({i}) << std::endl;
    }

    return std::make_tuple(im, targets);
}
#endif