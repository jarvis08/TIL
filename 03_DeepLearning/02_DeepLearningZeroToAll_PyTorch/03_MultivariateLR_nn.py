import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class MultivariateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.linear(x)
    

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = MultivariateModel()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

epochs = 10000
for epoch in range(1, epochs + 1):
    prediction = model(x_train)
    cost = F.mse_loss(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch: {}/{}, Prediction: {}, Cost: {:0.6f}'.format(
            epoch, epochs, prediction.squeeze().detach(), cost.item()
        ))
