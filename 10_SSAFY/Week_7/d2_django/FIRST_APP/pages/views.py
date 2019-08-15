from django.shortcuts import render
from django.http import HttpResponse
import random

# Create your views here.
def index(request):
    return render(request, 'index.html')

def home(request):
    name = '조동빈'
    data = ['강동주', '김지수', '정의진']
    empty_dataset = ['UBD', '명량', '성냥팔이 소녀의 재림']
    num = 10

    context = {
        'myname' : name,
        'class' : data,
        'empty_dataset' : empty_dataset,
        'num' : num,
    }
    # flask는 jinja를 사용하며
    # django는 dtl이라는 template engine을 사용
    return render(request, 'home.html', context)

def lotto(request):
    nums = sorted(random.sample(range(1,46), 6))
    context = {
        'lotto': nums,
    }
    return render(request, 'lotto.html', context)

def cube(request, num):
    result = num ** 3
    context = {
        'result': result,
    }
    return render(request, 'cube.html', context)

def match(request):
    goonghap = random.randint(50, 100)
    me = request.POST.get('me')
    you = request.POST.get('you')
    path_1 = request.path_info
    path_2 = request.path
    scheme = request.scheme
    method = request.method
    host = request.get_host
    context = {
        'me': me,
        'you': you,
        'goonghap': goonghap,
        'path_1': path_1,
        'path_2': path_2,
        'scheme': scheme,
        'method': method,
        'host': host,        
    }
    return render(request, 'match.html', context)