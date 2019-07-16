# Numpy

---

```python
import numpy as np

x=1
y=2
print(x+y)

npArray=np.array([1,4,2,5,3])
#pythonlist는sting,float등을다수보유가능
#numpy배열은통일됨.npArray중실수가섞을경우,배열의type은float
print('np.array::',npArray,'\n')

npArray=np.array([range(i,i+3)foriin[2,4,6]])
print('np.array::','\n',npArray,'\n')


#################################################난수배열생성
#재현가능을위한시드값지정
np.random.seed(0)

#0과1사이의난수
npRandom=np.random.random((3,3))
print('np.random.random::',npRandom,'\n')

#정규분포(평균=0,표준편차=1)난수
npNormal=np.random.normal(0,1,(3,3))
print('np.random.normal::',npNormal,'\n')

#0과10사이의임의의정수
npRandint=np.random.randint(0,10,(3,3))
print('np.random.randint::',npRandint,'\n')


#################################################특정값으로배열생성
#0으로채우기
npZeros=np.zeros(10,dtype=float)
print('np.zeros::',npZeros,'\n')

#1로채우기
npOnes=np.ones((3,5),dtype=int)
print('np.ones::','\n',npOnes,'\n')

#단위행렬
npEye=np.eye(3)
print('np.eye::','\n',npEye,'\n')

#지정값으로채우기
npFull=np.full((3,5),3.14)
print('np.full::','\n',npFull,'\n')


#################################################규칙에따라배열생성
npArange=np.arange(0,20,2)
print('np.arange::',npArange,'\n')

npLinspace=np.linspace(0,1,5)
print('np.linspace::',npLinspace,'\n')


#################################################배열속성탐색
x=np.random.randint(10,size=(3,4))

#ndim::차원의개수
#shape::각차원의크기
#size::전체배열크기
#dtype::데이터의타입
#itemsize::배열요소의크기를바이트단위로출력
#nbytes::배열의전체크기를바이트단위로출력
print('x::','\n',x)
print('xndim::',x.ndim)
print('xshape::',x.shape)
print('xsize::',x.size)
print('xdtype::',x.dtype)
print('xitemsize::',x.itemsize)
print('xnbytes::',x.nbytes)


#################################################배열슬라이싱
#x[start:stop:step]1
x=np.arange(10)
print('x::',x)

print('x[:5]::',x[:5])
print('x[5:]::',x[5:])
print('x[4:7]::',x[4:7])
print('x[::2]::',x[::2])

#1부터시작,하나걸러하나씩나열
print('x[1::2]::',x[1::2])

#모든요소역순으로나열
print('x[::-1]::',x[::-1])

#5부터시작,하나걸러역순으로나열
print('x[5::-2]::',x[5::-2])


#################################################다차원배열슬라이싱
x=np.random.randint(0,20,(3,4))
print('x::','\n',x)

#2행까지,3열까지
print('x[:2,:3]::','\n',x[:2,:3])

#2행까지,하나열걸러하나열씩
print('x[:2,::2]::','\n',x[:2,::2])

#역순행,역순열로
print('x[::-1,::-1]::','\n',x[::-1,::-1])

#첫번째열
print('x[:,0]::',x[:,0])

#첫행
print('x[0,:]::',x[0,:])
print('x[0]::',x[0])


#################################################넘파이는뷰(view)를슬라이싱
#파이썬리스트슬라이싱은사본(copy)를반환
#넘파이배열슬라이싱은뷰(view)를반환
#따라서넘파이배열을이용해,큰데이터의일부를수정하는등의행위편리
print('x::','\n',x)

x_sub=x[:2,:2]
print('x_sub::','\n',x_sub)

x_sub[0,0]=77
print('x_sub::','\n',x_sub)
print('x::','\n',x)

#넘파이로사본(copy)만들기
x_copy=x[:2,:2].copy()
print('x_copy::','\n',x_copy)

x_copy[0,0]=0
print('x_copy::','\n',x_copy)
print('x::','\n',x)


#################################################배열재구조화
npReshape=np.arange(1,10).reshape((3,3))
print('np.arange().reshape::','\n',npReshape)

#newaxis차원추가
x=np.arange(1,10).reshape((3,3))
print('x::','\n',x)
npNewaxis=x[np.newaxis,:]
print('[np.newaxis,:]::','\n',npNewaxis)
npNewaxis=x[:,np.newaxis]
print('[:,np.newaxis]::','\n',npNewaxis)


#################################################배열연결
x=np.arange(0,10,1)
print('x::',x)
y=np.arange(10,20,1)
print('y::',y)
z=np.concatenate([x,y])
print('z::',z)
#수직으로연결
print(np.vstack([x,y]))
#수평으로연결
print(np.hstack([x,y]))
#분할하기
x1,x2,x3=np.split(x,[3,6])
print('x1::',x1)
print('x2::',x2)
print('x3::',x3)
print('\n')
x=np.random.randint(0,20,(4,4))
print('x::','\n',x)
#수직으로분할
upper,lower=np.vsplit(x,[2])
print(upper)
print(lower)
print('\n')
left,right=np.hsplit(x,[2])
print(left)
print(right)


#################################################유니버셜함수
#np.add::+
#np.subtract::-
#np.negative::-
#np.multiply::*
#np.divide::/
#np.floor_divide몫연산:://
#np.power지수연산::**
#np.mod나머지연산::%
#np.abs
#np.pi파이(3.14)값
#np.log
#np.log2밑이2인log
#np.exp
#np.exp1()매우작은값계산할때
#np.log1p()매우작은값계산할때
#special.gamma
#special.gammaln
#special.beta
#special.erf
#special.erfc
#special.erfinv


#################################################출력지정
x=np.arange(5)
print(x)
y=np.empty(5)
print(y)
np.multiply(x,10,out=y)
print(y)

#띄워서지정
y=np.zeros(10)
print(y)
np.power(2,x,out=y[::2])
print(y)


#################################################집계
x=np.arange(1,6)
y=np.add.reduce(x)
#y=15

#곱하기는np.multiply.reduce(x)
#중간결과모두저장하려면np.multiply.accumulate(x)
#외적::np.multiply.outer(x,x)
#외적은두입력값의모든가능한쌍의곱을진행

#np.sum()
#np.min,max
#열의최소값을하려면axis=0,행의최소값은axis=1

#0을기준으로중앙정렬하기!
#x.mean(0)으로평균값을구한후
#전체배열에x.mean(0)값을빼서0을기준으로normalize됨


#################################################비교연산자
#np.equal::==
#np.not_equal::!=
#np.less::<
#np.less_equal::<=
#np.greater::>
#np.greater_equal::>=

x=np.array([1,2,3,4,5])
y=x<3
print(y)
y=x>3
print(y)
y=(2*x)==(x**2)
print(y)

#6보다작은거몇개?
print('3보다작은값개수::',np.count_nonzero(x<6))
print('3보다작은값개수::',np.sum(x<6))
print('2보다크고5보다작은값개수::',np.sum((x>2)&(x<5)))
print('4보다큰값개수::',np.sum(~((x<3)|(x<5))))

#np.biwise_and::&
#np.biwise_or::|
#np.biwise_xor::^
#np.biwise_not::~


#################################################팬시인덱싱
rand=np.random.RandomState(42)
x=rand.randint(100,size=10)
print('x::',x)

idx=[0,3,5]
print('[x[0].x[3],x[5]]::',x[idx])

x=np.arange(12).reshape((3,4))
print('x::','\n',x)

#순서대로row와col에해당하는값출력
row_idx=np.array([0,1,2])
col_idx=np.array([2,1,3])
print('row,cal::','\n',x[row_idx,col_idx])

#브로드캐스팅적용
print('x[row[:,np.newaxis],col]::','\n',x[row_idx[:,np.newaxis],col_idx])
print('row_idx[:,np.newaxis]*col_idx::','\n',row_idx[:,np.newaxis]*col_idx)

#결합인덱싱
print('x[2,[2,0,1]]::',x[2,[2,0,1]])


#################################################정렬
#정렬방법::삽입정렬(insertionsorts),선택정렬(selectionsort),병합정렬(mergesort),퀵정렬(quicksort),버블정렬(bubblesort)등
#np.sort는기본적으로퀵정렬알고리즘,병합과힙(heapsort)도사용가능
#정렬된인덱스를배열로반환하는건np.argsort(x)
#np.partition(x,k)는가장작은k값들을왼쪽에정렬한후나머지는원래대로출력


#################################################구조화된배열(복합적인이종데이터저장)
#구조화된배열+레코드배열
#PandasDataFrame사용시도움
#name,age,weight에대한세개의배열생성하지말아라!
#(dictionary형태로배열을만드는법)

name=['Alice','Bob','Cathy']
age=[25,45,37]
weight=[55.0,85.5,68.0]
#이세배열의연관성을나타내보자!(연결)

#dictionary형태로만들기!
data=np.zeros(3,dtype={'names':('name','age','weight'),'formats':('U10','i4','f8')})
#name,age,weight라는세가지배열
#위세가지배열은각각U10(최대길이10의유니코드문자열)
#i4(4바이트(32비트)정수)
#f8(8바이트,64비트부동소수점)의dtype을가짐

data['name']=name
data['age']=age
data['weight']=weight
print(data)

#데이터의전체이름가져오기
print(data['name'])

#데이터의첫번째행가져오기
print(data[0])

#마지막이름가져오기
print(data[-1]['name'])

#나이가30이하인이름가져오기
print(data[data['age']<30]['name'])

```

