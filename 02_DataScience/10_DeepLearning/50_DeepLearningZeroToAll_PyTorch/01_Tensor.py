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

## View
print('\n>> View')
# Numpy의 Reshape과 동일 기능
# shape [2, 2, 3]
np_3D = np.array([[[0, 1, 2],
                    [3, 4, 5]],

                    [[6, 7, 8],
                    [9, 10, 11]]])
t = torch.FloatTensor(np_3D)
print(t.shape)
# 2D로 축소해 보기, 1 차원에서는 3개의 원소를 가지며, 2 차원에서는 알아서(-1)
print(t.view([-1, 3]))
print(t.view([-1, 3]).shape)
# 같은 3D이지만, 다른 shape
print(t.view([-1, 1, 3]))
print(t.view([-1, 1, 3]).size())

## Squeeze
print('\n>> Squeeze')
# View와 동일한 역할이지만, Dimension의 element의 개수가 1일 경우, 그 dimension을 삭제
# shape [3, 1]
t = torch.FloatTensor([[0], [1], [2]])
print(t)
print(t.shape)
print(t.squeeze().shape)
print('dim=0: ', t.squeeze(dim=0).shape)
print('dim=0: ', t.squeeze(0).shape)
print(t.squeeze(0))
print('dim=1: ', t.squeeze(dim=1).shape)
print('dim=1: ', t.squeeze(1).shape)
print(t.squeeze(1))

## Unsqueeze
print('\n>> Unsqueeze')
t = torch.FloatTensor([1, 2, 3])
print(t)
print(t.shape)
print(t.squeeze().shape)
print('dim=0: ', t.unsqueeze(0).shape)
print(t.unsqueeze(0))
print('dim=1: ', t.unsqueeze(1).shape)
print(t.unsqueeze(1))

## Type Casting
print('\n>> TypeCasting')
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())
# Boolean
bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())

## Concatenate
print('\n>> Concatenate')
x = torch.FloatTensor([[1, 2], [3, 4]]) # [2, 2]
y = torch.FloatTensor([[5, 6], [7, 8]]) # [2, 2]
print(torch.cat([x, y], dim=0)) # [4, 2]
print(torch.cat([x, y], dim=1)) # [2, 4]

## Stacking
print('\n>> Stacking')
x = torch.FloatTensor([1, 4]) # [2,]
y = torch.FloatTensor([2, 5]) # [2,]
z = torch.FloatTensor([3, 6]) # [2,]
print(torch.stack([x, y, z])) # [3, 2]
print(torch.stack([x, y, z], dim=1)) # [2, 3]
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)]))
# [1, 2], [1, 2], [1, 2]

## Ones and Zeros
print('\n>> Ones and Zeros')
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)
print(torch.ones_like(x))
print(torch.zeros_like(x))
# multiple device를 사용하여 연산할 때, 다른 device의 변수와 계산을 하려 하면 에러 발생
# 만약 x를 기반으로 변수를 선언하면 x가 위치한 device에서 변수를 생성
# 따라서 device 차이로 인한 에러를 방지할 수 있다.

## In-place Operation
print('\n>> In-place Operation')
# 추가적인 메모리 사용 없이, 기존 메모리에 그대로 결과를 삽입하는 연산
x = torch.FloatTensor([[1, 2], [3, 4]])
print(x.mul(2.))
print(x) # 그대로
print(x.mul_(2.))
print(x) # 곱한 값이 저장되어 있다.
# 기능이 존재하지만, python garbage collector의 자동적인 메모리 free 기능으로 인해
# 사실상 성능 차이는 거의 없다고 PyTorch 측에서 인정