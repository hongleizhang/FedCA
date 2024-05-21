import torch


class HSIC(object):

    def __init__(self, config, x, y, sigma_x=1, sigma_y=1, ):
        self.x = x
        self.y = y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.config = config

    def _pairwiseDistances(self, x):
        # x should be two dimension
        instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    def _GaussianKernelMatrix(self, x, sigma=1):
        pairwise_distance = self._pairwiseDistances(x)
        return torch.exp(-pairwise_distance / sigma)

    def computeHSIC(self):
        m, _ = self.x.shape  # batch size
        K = self._GaussianKernelMatrix(self.x, self.sigma_x)
        L = self._GaussianKernelMatrix(self.y, self.sigma_y)
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))

        # if self.config['use_cuda']:
        #     H = H.cuda()

        K = torch.subtract(K, torch.mean(K))
        L = torch.subtract(L, torch.mean(L))
        # hsic = torch.trace(torch.mm(L, K)) / ((m - 1) ** 2)
        hsic = torch.trace(K*L) / ((m - 1) ** 2)

        # if self.config['use_cuda']:
        #     hsic = hsic.cpu()

        # hsic = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return hsic