import torch.nn as nn #nn = neural networks
import torch
import torch.tensor as tensor

x_data = tensor([[1.0], [2.0], [3.0]])
y_data = tensor([[2.0], [4.0], [6.0]])


class Model(nn.Module): #class 형태의 모델은 항상 nn.Module을 상속받아야 하며, super(모델명, self).__init__()을 통해 실행
    def __init__(self): #Initialize = Model에 사용될 구성요소들을 정의 및 초기화
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x): #specify the connections = __init__에서 정의된 요소들을 연결
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) #SGD = Gradient Descent(경사 하강법)의 일종/lr = learning rate(학습률)
#model.parameters()를 호출하면 모델의 멤버인 nn.Linear 모듈의 학습 가능한 매개변수들이 포함된다
# Training loop
for epoch in range(500):
    # 1) Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)

    # 2) Compute and print loss
    loss = criterion(y_pred, y_data)
    print(f'Epoch: {epoch} | Loss: {loss.item()} ') #f-문자열 = 표현식을 {expression}형태로 작성해서 값 삽입

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# After training
hour_var = tensor([[4.0]])
y_pred = model(hour_var)
print("Prediction (after training)",  4, model(hour_var).data[0][0].item())