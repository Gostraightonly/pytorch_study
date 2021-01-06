from torch.utils.data import Dataset, DataLoader #미니 배치 학습, 데이터 suffle, 병렬 처리 등을 간단히 수행 가능하다. Dataset을 정의하고 이를 DataLoader에 전달하는 방식
from torch import from_numpy, tensor
import numpy as np

class DiabetesDataset(Dataset): #custom dataset
    """ Diabetes dataset."""

    # Initialize your data, download, etc.
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',
                        delimiter=',', dtype=np.float32) #delimiter:구분 문자, float32:32bit 사용
        self.len = xy.shape[0]
        self.x_data = from_numpy(xy[:, 0:-1])
        self.y_data = from_numpy(xy[:, [-1]])

    def __getitem__(self, index): #dataset에서 특정 1개의 샘플을 가져오는 함수
        return self.x_data[index], self.y_data[index]

    def __len__(self): #dataset의 길이. 즉, 총 샘플의 수를 적어주는 부분
        return self.len


dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32, #batch는 training의 작은 단위 batch_size가 32이므로 32번의 iteraion을 통해 1번의 epoch이 수행됨.
                          shuffle=True,
                          num_workers=0) #windows 환경에서는 num_workers 0으로 설정

for epoch in range(2):
    for i, data in enumerate(train_loader, 0): #enumerate는 열거 객체를 돌려주는 python 내장함수. enumerate(iterable, start = 0)의 형태로 사용하며 iterable은 시퀀스, iterator 또는 iteration을 지원하는 다른 객체여야 함.
        # get the inputs
        inputs, labels = data #[inputs(기준), labels(당뇨여부)]의 목록인 data로부터 입력을 받는다.

        # wrap them in Variable(tensor로 설정)
        inputs, labels = tensor(inputs), tensor(labels)

        # Run your training process
        print(f'Epoch: {i} | Inputs {inputs.data} \n | Labels {labels.data}')
