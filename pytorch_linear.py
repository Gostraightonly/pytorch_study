import numpy as np #과학계산 라이브러리(다차원 배열 처리 유용)
import matplotlib.pyplot as plt #플롯(그래프) 그릴 때 사용
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w=1.0 #a random guess: random value

#our model for the forward pass
def forward(x):
	return x*w
def loss(x,y):
	y_pred = forward(x)
	return (y_pred-y)*(y_pred-y)

w_list = []
mse_list = []
for w in np.arange(0.0,4.1,0.1):
	print("w=",w)
	l_sum = 0
	for x_val, y_val in zip(x_data,y_data): #zip은 동일한 개수로 이루어진 자료형을 묶어줌
		y_pred_val = forward(x_val)
		l=loss(x_val,y_val)
		l_sum +=l
		print("\t",x_val,y_val,y_pred_val,l)
	print("MSE=",l_sum/3)
	w_list.append(w) #append는 리스트에 요소 추가
	mse_list.append(l_sum/3)

plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()
