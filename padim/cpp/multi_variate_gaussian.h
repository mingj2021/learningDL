#ifndef MULTI_VARIATE_GAUSSIAN_H
#define MULTI_VARIATE_GAUSSIAN_H

#include <torch/torch.h>

using torch::indexing::Slice;
using torch::nn::Module;
using torch::nn::ModuleHolder;
namespace F = torch::nn::functional;

struct MultiVariateGaussianImpl : Module
{
    MultiVariateGaussianImpl(int n_features, int n_patches)
    {
        register_buffer("mean", torch::zeros({n_features, n_patches}));
        register_buffer("inv_covariance", torch::eye(n_features).unsqueeze(0).repeat({n_patches, 1, 1}));
    }

    torch::Tensor _cov(torch::Tensor observations, bool rowvar, bool bias = false,
                       c10::optional<int> ddof = c10::nullopt, c10::optional<torch::Tensor> aweights = c10::nullopt)
    {
        // ensure at least 2D
        if (observations.dim() == 1)
            observations = observations.view({-1, 1});

        // treat each column as a data point, each row as a variable
        if (rowvar && observations.size(0) != 1)
            observations = observations.t();

        if (!ddof.has_value())
        {
            if (!bias)
                ddof = 1;
            else
                ddof = 0;
        }

        torch::Tensor weights, weights_sum, avg, observations_m, x_transposed;
        int fact = 0;
        if (aweights.has_value())
        {
        }
        else
        {
            avg = torch::mean(observations, 0);
            fact = observations.size(0) - ddof.value();
            observations_m = observations.sub(avg.expand_as(observations));
            x_transposed = observations_m.t();
        }
        auto covariance = torch::mm(x_transposed, observations_m);
        covariance = covariance / fact;
        return covariance.squeeze();
    }

    std::tuple<torch::Tensor,torch::Tensor> forward(torch::Tensor embedding)
    {
        auto device = embedding.device();
        int batch = embedding.size(0);
        int channel = embedding.size(1);
        int height = embedding.size(2);
        int width = embedding.size(3);

        auto embedding_vectors = embedding.view({batch, channel, height * width});

        auto dict = named_buffers(false);
        auto mean = dict["mean"];
        mean.copy_(torch::mean(embedding_vectors, {0}));
        auto covariance = torch::zeros({channel, channel, height * width}, device);
        auto identity = torch::eye(channel, device);
        for (int i = 0; i < height * width; i++)
        {
            /* code */
            torch::Tensor v = _cov(embedding_vectors.index({Slice(), Slice(), i}), false) + 0.01 * identity;
            covariance.index_put_({Slice(), Slice(), i}, v);
        }
        covariance = torch::linalg::inv(covariance.permute({2, 0, 1}));
        auto inv_covariance = dict["inv_covariance"];
        inv_covariance.copy_(covariance);
        return std::make_tuple(mean, inv_covariance);
    }

    std::tuple<torch::Tensor,torch::Tensor> fit(torch::Tensor embedding)
    {
        return forward(embedding);
    }
};
TORCH_MODULE(MultiVariateGaussian);
#endif