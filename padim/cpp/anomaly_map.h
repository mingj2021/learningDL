#ifndef ANOMALY_MAP_H
#define ANOMALY_MAP_H

#include <torch/torch.h>

using torch::indexing::Slice;
using torch::nn::Module;
using torch::nn::ModuleHolder;
namespace F = torch::nn::functional;

torch::Tensor gaussian(int window_size, float sigma, c10::TensorOptions options = {})
{
    auto x = torch::arange(window_size, options) - int(window_size / 2);
    if (window_size % 2 == 0)
        x = x + 0.5;
    auto gauss = torch::exp(-(x * x) / (2 * sigma * sigma));
    return gauss / gauss.sum();
}

torch::Tensor get_gaussian_kernel1d(int window_size, float sigma, c10::TensorOptions options = {})
{
    auto window_1d = gaussian(window_size, sigma, options);
    return window_1d;
}

torch::Tensor get_gaussian_kernel2d(std::tuple<int, int> kernel_size, std::tuple<float, float> sigma, c10::TensorOptions options = {})
{
    auto [ksize_x, ksize_y] = kernel_size;
    auto [sigma_x, sigma_y] = sigma;
    auto kernel_x = get_gaussian_kernel1d(ksize_x, sigma_x, options);
    auto kernel_y = get_gaussian_kernel1d(ksize_y, sigma_y, options);
    auto kernel_2d = torch::matmul(kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t());
    return kernel_2d;
}

std::vector<long int> _compute_padding(std::vector<long int> kernel_size)
{
    int lens = kernel_size.size();
    std::vector<long int> computed, out_padding;
    computed.resize(lens);
    for (int i = 0; i < lens; i++)
    {
        computed[i] = kernel_size[i] -1;
    }
    
    out_padding.resize(lens * 2);
    for (int i = 0; i < lens; i++)
    {
        /* code */
        auto computed_tmp = computed[lens - 1 - i];
        auto pad_front = int(computed_tmp / 2);
        auto pad_rear = computed_tmp - pad_front;

        out_padding[2 * i + 0] = pad_front;
        out_padding[2 * i + 1] = pad_rear;
    }
    return out_padding;
}

torch::Tensor normalize_kernel2d(torch::Tensor input)
{
    torch::Tensor norm = input.abs().sum(-1).sum(-1);
    return input / norm.unsqueeze(-1).unsqueeze(-1);
}

int compute_kernel_size(float sigma_val)
{
    return 2 * int(4.0 * sigma_val + 0.5) + 1;
}

struct GaussianBlur2dImpl : Module
{
    GaussianBlur2dImpl(std::tuple<float, float> sigma, std::tuple<int, int> kernel_size, int channels = 1,
                       bool normalize = true) :
                        m_channels(channels)
    {
        torch::Tensor m_kernel;
        m_kernel = get_gaussian_kernel2d(kernel_size, sigma);
        if (normalize)
            m_kernel = normalize_kernel2d(m_kernel);
        m_kernel.unsqueeze_(0).unsqueeze_(0);
        m_kernel.expand({m_channels, -1, -1, -1});
        register_buffer("kernel", m_kernel);
        m_height = m_kernel.size(2);
        m_width = m_kernel.size(3);
        padding_shape = _compute_padding({m_height, m_width});
    }

    torch::Tensor forward(torch::Tensor input_tensor)
    {
        auto dict = named_buffers(false);
        auto m_kernel = dict["kernel"];
        std::cout << m_kernel.sizes() << std::endl;
        int batch = input_tensor.size(0);
        int channel = input_tensor.size(1);
        int height = input_tensor.size(2);
        int width = input_tensor.size(3);

        input_tensor = F::pad(input_tensor, F::PadFuncOptions(padding_shape).mode(torch::kReflect));
        std::cout << input_tensor.sizes() << std::endl;
        auto output = F::conv2d(input_tensor, m_kernel, F::Conv2dFuncOptions().groups(m_channels).padding(0).stride(1));
        std::cout << output.sizes() << std::endl;
        auto out = output.view({batch, channel, height, width});
        std::cout << out.sizes() << std::endl;
        return out;
    }

    int m_channels, m_height, m_width;
    std::vector<long int> padding_shape;
};
TORCH_MODULE(GaussianBlur2d);


struct AnomalyMapGeneratorImpl : Module
{
    AnomalyMapGeneratorImpl(std::tuple<int, int> image_size, float sigma):
    m_image_size(image_size)
    {
        auto kernel_size = 2 * int(4.0 * sigma + 0.5) + 1;
        m_blur = GaussianBlur2d(std::tuple<float, float>(sigma, sigma), std::tuple<int, int>(kernel_size, kernel_size));
        register_module("m_blur", m_blur);
    }
    
    torch::Tensor compute_distance(torch::Tensor embedding, torch::Tensor mean, torch::Tensor inv_covariance)
    {
        int batch = embedding.size(0);
        int channel = embedding.size(1);
        int height = embedding.size(2);
        int width = embedding.size(3);

        embedding = embedding.reshape({batch, channel, height * width});
        auto delta = (embedding - mean).permute({2, 0, 1});

        auto distances = (torch::matmul(delta, inv_covariance) * delta).sum(2).permute({1, 0});
        distances = distances.reshape({batch, 1, height, width});
        distances = distances.clamp(0).sqrt();

        return distances;
    }

    torch::Tensor up_sample(torch::Tensor distance)
    {
        auto [height, width] = m_image_size;
        auto score_map = F::interpolate(distance, F::InterpolateFuncOptions().size(std::vector<int64_t>({height, width})).mode(torch::kBilinear).align_corners(false));
        return score_map;
    }

    torch::Tensor smooth_anomaly_map(torch::Tensor anomaly_map)
    {
        auto blurred_anomaly_map = m_blur(anomaly_map);
        return blurred_anomaly_map;
    }

    torch::Tensor compute_anomaly_map(torch::Tensor embedding, torch::Tensor mean, torch::Tensor inv_covariance)
    {
        auto score_map = compute_distance(embedding, mean, inv_covariance);
        auto up_sampled_score_map = up_sample(score_map);
        auto smoothed_anomaly_map = smooth_anomaly_map(up_sampled_score_map);

        return smoothed_anomaly_map;

    }

    torch::Tensor forward(torch::Tensor embedding, torch::Tensor mean, torch::Tensor inv_covariance)
    {
        auto anomaly_map = compute_anomaly_map(embedding, mean, inv_covariance);
        return anomaly_map;
    }
    std::tuple<int, int> m_image_size;
    GaussianBlur2d m_blur=nullptr;
};
TORCH_MODULE(AnomalyMapGenerator);
#endif