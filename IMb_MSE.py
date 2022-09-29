import numpy as np
import torch
from torch.distributions import MultivariateNormal as MVN
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture

y_train = y_a1_train       #  prior of the training data
def get_gmm(dist, n_components):
    # fit a **ground truth** label distribution
    all_labels = torch.tensor(dist)     # assume sufficient samples
    if len(all_labels.shape) == 1:
        all_labels = all_labels.unsqueeze(-1)
    gmm = GaussianMixture(n_components=n_components).fit(all_labels)
    gmm_dict = {'means': gmm.means_, 'weights': gmm.weights_, 'variances': gmm.covariances_}
    return gmm_dict

def gradient_gai(y_pred, y_true):
    gmm = get_gmm(dist= y_train, n_components=6 #if TRAIN_DIST == 'lognormal' else 64
                 )
    gmm = {k: torch.tensor(gmm[k]) for k in gmm}
    noise_var = 1.
    target, pred = torch.tensor(y_true).reshape(-1,1), torch.tensor(y_pred).reshape(-1,1).requires_grad_()
    I = torch.eye(target.shape[-1])
    mse_term = -MVN(target, noise_var*I).log_prob(pred)
    balancing_term = MVN(gmm['means'], gmm['variances']+noise_var*I).log_prob(pred.unsqueeze(1)) + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=1)
    loss = -mse_term - balancing_term
    loss = (loss * (2 * noise_var)).sum()
    loss.backward()
    grad_pred = pred.grad
    grad_pred = np.array(grad_pred).reshape(-1,1)
    return grad_pred

def hessian_gai(y_pred, y_true):
    y_true, y_pred = np.array(y_true).reshape(-1,1), np.array(y_pred).reshape(-1,1)
    return 0*(y_true + y_pred) + 1

def IMb_loss3(y_pred, y_true):
    grad = gradient_gai(y_pred, y_true)
    hess = hessian_gai(y_pred, y_true)
    return grad, hess

if __name__ == '__main__':
    main()

