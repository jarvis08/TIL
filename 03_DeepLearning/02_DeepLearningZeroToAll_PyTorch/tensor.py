import torch
import numpy as np

## Numpy 1D
print('\n>> Numpy 1D')
np_1D = np.array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
print('Rank  of 1D:', np_1D.ndim)
print('Shape of 1D:', np_1D.shape)

## Numpy 2D
print('\n>> Numpy 2D')
np_2D = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9]])
print('Rank  of 2D:', np_2D.ndim)
print('Shape of 2D:', np_2D.shape)
print('\nnp_2D[1:3, :-1]')
print(np_2D[1:3, :-1])

## Tensor 1D
print('\n>> Tensor 1D')
tensor_1D = torch.FloatTensor([1., 2., 3., 4., 5., 6., 7., 8., 9.])
print(tensor_1D.dim()) # Rank
print(tensor_1D.shape) # Shape
print(tensor_1D.size()) # Size
print(tensor_1D[1]) # Indexing
print(tensor_1D[5:-1]) # Slicing

## Tensor 2D
print('\n>> Tensor 2D')
tensor_2D = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9]])
print(tensor_2D.dim()) # Rank
print(tensor_2D.shape) # Shape
print(tensor_2D.size()) # Size
print('\ntensor_2D[:, 1]')
print(tensor_2D[:, 1]) # 첫 번째 차원 전부, 두 번째 차원 1번 인덱스만
print(tensor_2D[:, 1].size())
print('\ntensor_2D[:, :-1]')
print(tensor_2D[:, :-1])
print(tensor_2D[:, :-1].size())
print('\ntensor_2D[1:3, :]')
print(tensor_2D[1:3, :])
print(tensor_2D[1:3, :].shape)

## Broadcasting
print('\n>> Broadcasting')
# Same Shape
t1 = torch.FloatTensor([[2, 3]])
t2 = torch.FloatTensor([[2, 3]])
print('Same Shape: ', t1 + t2)
# Scalar + Vector
t1 = torch.FloatTensor([[3]])
t2 = torch.FloatTensor([[2, 3]])
print('Scalar + Vecotr: ', t1 + t2)
# (2x1 Vector) + (1x2 Vector)
# 각각을 2x2로 변형하여 계산
t1 = torch.FloatTensor([[1, 2]])
t2 = torch.FloatTensor([[3], [4]])
print('2x1 Vector + 1x2 Vector:\n', t1 + t2)

## Multiplication
# Matrix Multiplication, 행렬곱
m1 = torch.FloatTensor([[1, 2], [3, 4]]) # 2x2
m2 = torch.FloatTensor([[1], [2]]) # 2x1
print('>> Matrix Multiplication, m1.matmul(m2):\n', m1.matmul(m2)) # 2x1
print('\n')
# Broadcasting + Multiplication
m1 = torch.FloatTensor([[1, 2], [3, 4]]) # 2x2
m2 = torch.FloatTensor([[1], [2]]) # 2x1
# 2를 2x2로 변환하여 요소별 곱
# m2 => [[1, 1], [2, 2]]
print('>> Broadcasting + Multiplication, m1*m2, m1.mul(m2):')
print(m1 * m2)
print(m1.mul(m2))

## Mean
print('>> Mean')
# Numpy와 유사하지만, Dimension argument의 명칭이 numpy에서는 axis, PyTorch는 dim
# mean()은 integer에 대해서는 불가하므로, LongTensor()로는 불가
t = torch.FloatTensor([1, 2])
print('t.mean() = ', t.mean())
t = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('[[1, 2, 3], [4, 5, 6], [7, 8, 9]]')
print(t.mean()) # 전체 요소들 평균
print('dim= 0: ', t.mean(dim=0))
print('dim= 1: ', t.mean(dim=1))
print('dim=-1: ', t.mean(dim=-1))

## Sum
print('\n>> Sum')
# t.sum() 형태, mean()과 동일한 형태
print('[[1, 2, 3], [4, 5, 6], [7, 8, 9]]')
print(t.sum()) # 전체 요소들 평균
print('dim= 0: ', t.sum(dim=0))
print('dim= 1: ', t.sum(dim=1))
print('dim=-1: ', t.sum(dim=-1))

## Max & Argmax
print('\n>> Max & Argmax')
# argmax는 max 값의 index를 반환
print('t.max() = ', t.max())
print('t.argmax() = ', t.argmax()) # 하나의 배열에서 찾은 것 처럼 인덱싱
print(t.max(dim=0))
print(t.max(dim=0)[0])
print(t.max(dim=0)[1])
print(t.argmax(dim=0)) # argmax()는 max()[1]과 동일하며, max() 만으로도 확인 가능
