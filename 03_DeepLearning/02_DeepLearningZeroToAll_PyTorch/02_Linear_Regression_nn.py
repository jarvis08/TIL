import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.linear(x)


x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[3], [5], [7]])

# Linear Regression에서는 MSE 사용
# l1_loss, smooth_l1_loss
model = LinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

epochs = 100
for epoch in range(epochs + 1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad() # gredient 초기화
    cost.backward() # gredient 계산
    optimizer.step() # gredient 적용
    print('Epoch {:4d}/{} Cost: {:.6f}'.format(
        epoch, epochs, cost.item()
    ))