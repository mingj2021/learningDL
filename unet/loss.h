#ifndef LOSS_H
#define LOSS_H

#include <torch/torch.h>
namespace F = torch::nn::functional;

torch::Tensor DiceLoss(torch::Tensor input, torch::Tensor target,int cls=15, float epsilon = 1e-6)
{
    input = F::softmax(input, F::SoftmaxFuncOptions(1)).to(torch::kFloat);
    target = F::one_hot(target, cls).permute({0, 3, 1, 2}).contiguous().to(torch::kFloat);
    
    
    input = input.flatten(1);
    target = target.flatten(1);
    auto inter = (input * target).sum(1);
    auto loss = 1 - ((2 * inter + epsilon) / (input.sum(1) + target.sum(1)));
    return loss.mean();
}

torch::Tensor dice_coeff(torch::Tensor input, torch::Tensor target, bool reduce_batch_first=false, float epsilon=1e-6)
{
    std::vector<int64_t> vec;
    if(input.dim() == 2 || ! reduce_batch_first)
    {
        vec = { -1,-2 };
    }
    else
    {
        vec = { -1,-2,-3 };
    }
    auto sum_dim = c10::makeArrayRef(vec);
    auto inter = 2 * (input * target).sum(sum_dim);
    auto sets_sum = input.sum(sum_dim) + target.sum(sum_dim);
    // std::cout << inter << std::endl;
    // std::cout << sets_sum << std::endl;
    // sets_sum = torch::where( sets_sum == 0, inter, sets_sum);
    auto dice = (inter + epsilon) / (sets_sum + epsilon);
    return dice.mean();
}

torch::Tensor multiclass_dice_coeff(torch::Tensor input, torch::Tensor target, bool reduce_batch_first=false, float epsilon=1e-6)
{
    return dice_coeff(input.flatten(0,1), target.flatten(0,1), reduce_batch_first, epsilon);
}

torch::Tensor dice_loss(torch::Tensor input, torch::Tensor target, bool multiclass = false)
{
    if(multiclass)
    {
        return 1 - multiclass_dice_coeff(input, target, true);
    }
    else
    {
        return 1- dice_coeff(input, target, true);
    }

}
#endif