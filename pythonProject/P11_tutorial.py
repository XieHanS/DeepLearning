# 第一个神经网络模型
import torch
from torch.autograd import Variable as V
import torch.nn.functional as F
import matplotlib.pylab as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = V(x), V(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        # 接下来就是自己的模型
        self.hidden = torch.nn.Linear(n_feature, n_hidden, n_output)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# 搭模型
##
# Net的第一个参数表示一个特征向量包含的特征数，
# Net的第二个参数表示隐藏层的神经元个数
# Net的第三个参数表示输出层的类型个数，二分类[0,1]或者[1,0]
##
net = Net(1, 10, 1)  # 输入一个元素,隐藏层10个神经元，输出一个
print(net)

plt.ion()  # 实时画图的过程
plt.show()

# 优化模型
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)  # 优化器所有参数，学习效率
loss_func = torch.nn.MSELoss()  # 损失函数，计算误差的手段，均方误差

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)  # 预测值在前

    optimizer.zero_grad()  # 梯度先设为0
    loss.backward()
    optimizer.step()

    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()


