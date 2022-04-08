import torch
from torch._C import device
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


def l2_normalize(r):
    r_reshape = r.reshape(r.shape[0], -1, 1, 1)
    r_normalize = torch.norm(r_reshape, p=2, dim=1, keepdim=True)
    r = r / (r_normalize + 1e-9)
    return r


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VATLoss(nn.Module):
    def __init__(self, args):
        super(VATLoss, self).__init__()
        self.xi = args.vat_xi
        self.eps = args.vat_eps
        self.vat_iter = args.vat_iter

    def forward(self, model, x):
        model.eval()  # for switching off BN
        r = torch.normal(0, 1, size=x.shape)
        r = l2_normalize(r)
        x = x.to(device)
        outputs = model(x)
        log_sftmx = nn.LogSoftmax(dim=-1)
        kl_div = nn.KLDivLoss(reduction="batchmean", log_target=True)
        with torch.no_grad():
            outputs = log_sftmx(outputs)
        for i in range(self.vat_iter):
            r = r.to(device)
            r.requires_grad_()

            adv = x + self.xi * r
            adv_outputs = model(adv)
            adv_outputs = log_sftmx(adv_outputs)
            adv_distance = kl_div(adv_outputs, outputs)
            adv_distance.backward()
            r = r.grad
            r = l2_normalize(r)
            model.zero_grad()
        r_adv = r * self.eps
        model.train()
        return r_adv
