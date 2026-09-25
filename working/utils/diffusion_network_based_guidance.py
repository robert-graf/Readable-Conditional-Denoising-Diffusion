from functools import partial
from torch import Tensor
import torch

# Warning: This code may contains bugs and is not well tested


def network_based_guidance(cond_fn, x_t: Tensor, t: Tensor, variance: Tensor, x_conditional, label, embedding) -> Tensor:
    """
    In literature called classifier guidance, but we can do this with any regression task

    Parameters:
            cond_fn (fun):  returns grad(log(p(y|x)));
                            inputs are x_t, t, Optional[x_condition], Optional[label], Optional[embedding]
            x_t (Tensor): Sample after step t (batched)
            t (Tensor): timestep t (batched)
            variance: fixed or learned variance.
            Optional[x_condition],
            Optional[label],
            Optional[embedding]


    Returns:
            binary_sum (str): Binary string of the sum of a and b
    cond_fn():

    From the original code from
    https://github.com/openai/guided-diffusion/blob/main/scripts/classifier_sample.py
    Compute the mean for the previous step, given a function cond_fn that
    computes the gradient of a conditional log probability with respect to
    x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
    condition on y.
    This uses the conditioning strategy from Sohl-Dickstein et al. (2015), when using a classifier.
    """

    gradient = cond_fn(x_t=x_t, t=t, x_conditional=x_conditional, label=label, embedding=embedding)
    return variance * gradient.float()


def __cond_fn_classifier(classifier, scale, x_t=None, t=None, label=None, **kwargs) -> Tensor:
    assert label is not None
    # https://github.com/openai/guided-diffusion/blob/main
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        logits = classifier(x_in, t)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected = log_probs[range(len(logits)), label.view(-1)]
        return torch.autograd.grad(selected.sum(), x_in)[0] * scale


def get_cond_fn_classifier(classifier, scale):
    return partial(__cond_fn_classifier, classifier=classifier, scale=scale)


def __cond_fn_from_conditional_2_networks(
    conditional2regression,
    target2regression,
    scale,
    loss_fn=torch.nn.L1Loss(reduction="none"),
    x_t=None,
    t=None,
    x_condition=None,
    **kwargs
) -> Tensor:
    """
    Give two models that produce the same output, but one on target domain and one on condition domain.
    The x_t will be slightly moved to fit to the regression goal of the condition.

    To abstract? Lets make an example:
    You have as a condition a male face and want to generate a female face.
    You want to force the same eye color.
    You need two networks:
    1.) one trained on noisy (same as diffusion step 0 to T) female faces that predicts eye color;
    2.) the other is trained on male faces to predict eye color.
    We get the eye color by apply network 2 on the condition. We backpropagate through network 1,
    so the generated diffusion image will be the same eye-color as predicted.

    Any regression task works: Classification, Feature prediction (age, color, weight), Landmark prediction, Segmentation.
    """
    assert x_condition is not None
    # compute the same info on the condition.
    regression_from_condition = conditional2regression(x_condition).detach()
    with torch.enable_grad():
        x_in = x_t.detach().requires_grad_(True)
        # compute the output of network given the generated image
        regression_from_target = target2regression(x_in, t)

        # L1 Loss, that should be minimized (any loss works here.)
        loss = torch.log(loss_fn(regression_from_target, regression_from_condition))
        return torch.autograd.grad(loss.sum(), x_in)[0] * scale


from torch.autograd import Variable


def __cond_fn_from_any_network(
    wish_regression, target2regression, scale, loss_fn=torch.nn.L1Loss(reduction="none"), x_t=None, t=None, **kwargs
) -> Tensor:
    """
    Give a models that produce the same regression as wish_regression
    Any regression task works: Classification, Feature prediction (age, color, weight), Landmark prediction, Segmentation.
    """
    with torch.enable_grad():
        wish_regression = wish_regression.detach().requires_grad_(True)
        # t = t.detach().requires_grad_(True)
        x_in = x_t.detach().requires_grad_(True)
        # compute the same info on the condition.
        regression = target2regression(x_in, t)
        # L1 Loss, that should be minimized (any loss works here.)
        loss = torch.log(torch.abs(regression - wish_regression))
        # print(type(x_in), type(loss), type(regression), type(t), type(target2regression))
        a = torch.autograd.grad(loss.sum(), x_in)[0]
        # print(a)
        return a * scale


def get_cond_fn_from_conditional(a, target2regression, scale, fixed):
    if fixed:
        return partial(
            __cond_fn_from_any_network,
            wish_regression=a,
            target2regression=target2regression,
            scale=scale,
        )
    else:
        return partial(
            __cond_fn_from_conditional_2_networks,
            conditional2regression=a,
            target2regression=target2regression,
            scale=scale,
        )
