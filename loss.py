import torch
import pickle
import numpy as np


def logloss(Ptrue, Pred, szs, eps=10e-10):
    b, h, w, ch = szs
    Pred = torch.clip(Pred, eps, 1.)
    Pred = -torch.log(Pred)
    Pred = Pred * Ptrue
    Pred = torch.reshape(Pred, (b, h * w * ch))
    Pred = torch.sum(Pred, dim=1)
    return Pred


def l1(true, pred, szs):
    b, h, w, ch = szs
    res = torch.reshape(true - pred, (b, h * w * ch))
    res = torch.abs(res)
    res = torch.sum(res, dim=1)
    return res


# Ytrue: (32, 13, 13, 9)
# Ypred: (32, 13, 13, 8)
#
# Compare 2 batch of image matrices together (not the vertices)
# TODO: Questions:
# 1. Does the loss use unwarped or warped objects as the inputs?
# 2. Why does the code warp the identity square?
# 3. Why we need to use non-object loss? --> Verify when training
def loss(Y_true, Y_pred):
    b, w, h, _ = Y_true.shape

    obj_probs_true = Y_true[..., 0]
    obj_probs_pred = Y_pred[..., 0]

    non_obj_probs_true = 1. - Y_true[..., 0]
    non_obj_probs_pred = Y_pred[..., 1]

    # affine_pred = Y_pred[..., 1:]
    affine_pred = Y_pred[..., 2:]
    pts_true = Y_true[..., 1:]

    # define affine transformation in x-axis and y-axis
    affine_x = torch.stack([
        torch.maximum(affine_pred[..., 0], torch.tensor(0.)),  # v3
        affine_pred[..., 1],  # v4
        affine_pred[..., 2],  # v7
    ], dim=3)

    # v5, v6, v8
    affine_y = torch.stack([
        torch.maximum(affine_pred[..., 3], torch.tensor(0.)),  # v5
        affine_pred[..., 4],  # v6
        affine_pred[..., 5],  # v8
    ], dim=3)

    # generate identity rectangle
    v = 0.5
    # 1. mean plural
    base = torch.tensor([[[[-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.]]]])
    base = base.repeat(b, h, w, 1)

    pts = torch.zeros((b, h, w, 0), dtype=torch.float32)

    for i in range(0, 12, 3):
        row = base[..., i:(i + 3)]
        pts_x = torch.sum(affine_x * row, dim=3, keepdim=True)
        pts_y = torch.sum(affine_y * row, dim=3, keepdim=True)
        # TODO: why sum?

        pts_xy = torch.cat([pts_x, pts_y], dim=3)
        pts = torch.cat([pts, pts_xy], dim=3)

    flags = torch.reshape(obj_probs_true, (b, h, w, 1))
    res = 1. * l1(pts_true * flags, pts * flags, (b, h, w, 4 * 2))
    res += 1. * logloss(obj_probs_true, obj_probs_pred, (b, h, w, 1))
    res += 1. * logloss(non_obj_probs_true, non_obj_probs_pred, (b, h, w, 1))

    return res


if __name__ == "__main__":
    pred = np.load(open("test_pred.npy", "rb"))
    true = np.load(open("test_true.npy", "rb"))

    print(loss(torch.tensor(true), torch.tensor(pred)))
