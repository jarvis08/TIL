import torch
import torch.optim as optim

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
print(x_train.shape)
print(y_train.shape)
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W, b], lr=1e-5)

epochs = 10000
for epoch in range(1, epochs + 1):
    hypothesis = x_train.matmul(W) + b
    
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
        ))
        print('Hypothesis: ', hypothesis.squeeze().detach())

