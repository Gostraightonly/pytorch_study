from torch import nn, optim, from_numpy #from_numpy는 numpy 배열을 입력받아 tensor(dtype)로 바꿀 때 사용
import numpy as np

xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32) #np.loadtxt csv, txt 등의 파일 읽기
x_data = from_numpy(xy[:,0:-1]) #행렬 범위 지정
y_data = from_numpy(xy[:,[-1]])
print(f'X\'s shape: {x_data.shape} | Y\'s shape: {y_data.shape}')

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l1 = nn.Linear(8.6) #8개가 input
        self.l2 = nn.Linear(6,4) #중간 layer는 몇 개가 되어도 상관없음
        self.l3 = nn.Linear(4,1) #1개가 output

        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred

#out model
model = Model()

#criterion&optimizer
criterion = nn.BCELoss(reduction = 'mean')
optimizer = optim.SGD(model.parameters(),lr=0.1)

#training loop
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(f'Epoch : {epoch+1}/100 | Loss : {loss.item():.4f}')

    optimizer.zero_grad
    loss.backward()
    optimizer.step()

