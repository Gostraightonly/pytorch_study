import torch
from torch.autograd import Variable #autograd 패키지는 Tensor의 모든 연산에 대해 자동 미분을 제공

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = Variable(torch.Tensor([1.0]), requires_grad = True)  #Any random value
#torch.Tensor에서 requires_grad속성을 True로 설정하면 tensor에서 이뤄진 모든 연산들을 추적, .grad속성에 누적된다. 이후 .backward()를 이용하여 모든 gradient 자동계산.
#[1.0]은 초기값

#Our model forward pass
def forward(x):
    return x*w
#Loss function
def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y) * (y_pred-y)

#Before training
print("predict (before training)", 4, forward(4).data[0])

#Training loop
for epoch in range(10):
    for x_val, y_val in zip(x_data,y_data):
        l = loss(x_val,y_val)
        l.backward()
        print("\tgrad: ",x_val, y_val, w.grad.data[0])
        w.data = w.data - 0.01*w.grad.data

        #Manually zero the gradients after updating weights
        w.grad.data.zero_()

    print("progress:", epoch, l.data[0])

#After training
print("predict (after training)", 4, forward(4).data[0])
