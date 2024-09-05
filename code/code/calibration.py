import torch
import torch.nn as nn
class Identity(torch.nn.Module):
    def __init__(self): 
        super(Identity, self).__init__()

    def forward(self, x): 
        return x

    def predict(self, x):
        return x

class LogisticRegressionModel(torch.nn.Module): 
  
    def __init__(self): 
        super(LogisticRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)  # One in and one out 
  
    def forward(self, x):
        return self.linear(x)

def LRcalibrator(netG, netD_lis, data_loader, device, nz=100, calib_frac=0.1):
    n_batches = int(calib_frac * len(data_loader))
    batch_size = data_loader.batch_size
    # define a shortcut
    def gen_scores(batch_size):
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        x = netG(noise)
        # softmax combination
        softmax_1 = nn.Softmax(dim=0)

        D_lis1 = []
        for netD_tmp in netD_lis:
            D_lis1.append(netD_tmp(x))
        D_lis = torch.stack(D_lis1, 0)

        output_weight = softmax_1(D_lis)
        output_tmp = torch.mul(D_lis, output_weight)
        D_tmp = output_tmp.mean(dim=0)
        return D_tmp

    print('prepare real scores ...')
    scores_real = []
    for i, (data, _) in enumerate(data_loader):
        # softmax combination
        softmax_1 = nn.Softmax(dim=0)

        real_score1 = []
        for netD_tmp in netD_lis:
            real_score1.append(netD_tmp(data.to(device)))
        real_score = torch.stack(real_score1, 0)

        output_weight = softmax_1(real_score)
        output_tmp = torch.mul(real_score, output_weight)
        score_tmp = output_tmp.mean(dim=0)

        scores_real.append(score_tmp)
        if i > n_batches:
            break
    scores_real = torch.cat(scores_real, dim=0)
    print('prepare fake scores ...')
    scores_fake=gen_scores(batch_size)

    print('training LR calibrator ...')
    model = LogisticRegressionModel().to(device)
    x = torch.cat([scores_real, scores_fake], dim=0)
    y = torch.cat([torch.ones_like(scores_real),
                torch.zeros_like(scores_fake)], dim=0)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1)
    for epoch in range(5000):
        optimizer.zero_grad()
        with torch.enable_grad():
            pred_y = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, y)
            loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 1000 == 0: 
            print('Epoch: %d; Loss:%.3f' % (epoch, loss.item()))
    return model