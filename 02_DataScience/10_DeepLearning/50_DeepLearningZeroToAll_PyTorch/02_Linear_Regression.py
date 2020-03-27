import torch

epochs = 100
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [5], [7]])

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=0.01)

for epoch in range(1, epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train)**2)

    print('Epoch {:4d}/{} W: {:.3f} Cost: {:.6f}'.format(
        epoch, epochs, W.item(), cost.item()
    ))

    optimizer.zero_grad() # gredient 초기화
    cost.backward() # back-propagation & cost 및 gredient 계산
    optimizer.step() # gredient 적용