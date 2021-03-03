import torch
import pickle
import numpy as np


def logloss():
    pass


def l1():
    pass

# def logloss(Ptrue,Pred,szs,eps=10e-10):
# 	b,h,w,ch = szs
# 	Pred = tf.clip_by_value(Pred,eps,1.)
# 	Pred = -tf.log(Pred)
# 	Pred = Pred*Ptrue
# 	Pred = tf.reshape(Pred,(b,h*w*ch))
# 	Pred = tf.reduce_sum(Pred,1)
# 	return Pred
#
# def l1(true,pred,szs):
# 	b,h,w,ch = szs
# 	res = tf.reshape(true-pred,(b,h*w*ch))
# 	res = tf.abs(res)
# 	res = tf.reduce_sum(res,1)
# 	return res


# Ytrue: (32, 13, 13, 9)
# Ypred: (32, 13, 13, 8)
def loss(Y_true, Y_pred):
    b, w, h, _ = Y_true.shape

    obj_probs_true = Y_true[..., 0]
    obj_probs_pred = Y_pred[..., 0]

    affine_pred = Y_pred[..., 2:]
    pts_true = Y_true[..., 1:]

    # define affine transformation in x-axis and y-axis
    affine_x = torch.stack([
        torch.maximum(affine_pred[..., 0], torch.tensor(0.)),
        affine_pred[..., 1],
        affine_pred[..., 2],
    ], dim=3)

    affine_y = torch.stack([
        affine_pred[..., 3],
        torch.maximum(affine_pred[..., 4], torch.tensor(0.)),
        affine_pred[..., 5],
    ], dim=3)

    v = 0.5
    base = torch.tensor([[[[-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.]]]])
    base = base.repeat(b, h, w, 1)

    pts = torch.zeros((b, h, w, 0), dtype=torch.float32)

    for i in range(0, 12, 3):
        row = base[..., i:(i + 3)]
        pts_x = torch.sum(affine_x * row, dim=3, keepdim=True)
        pts_y = torch.sum(affine_y * row, dim=3, keepdim=True)

        pts_xy = torch.cat([pts_x, pts_y], dim=3)
        pts = torch.cat([pts, pts_xy], dim=3)

    flags = torch.reshape(obj_probs_true, (b, h, w, 1))
    res = 1. * l1(pts_true * flags, pts * flags, (b, h, w, 4 * 2))
    res += 1. * logloss(obj_probs_true, obj_probs_pred, (b, h, w, 1))

    return res


if __name__ == "__main__":
    pred = np.load(open("test_pred.npy", "rb"))
    true = np.load(open("test_true.npy", "rb"))

    loss(torch.tensor(true), torch.tensor(pred))
