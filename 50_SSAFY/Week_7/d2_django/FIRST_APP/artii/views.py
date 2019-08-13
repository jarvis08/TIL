from django.shortcuts import render
import requests


def artii(request):
    font_url = 'http://artii.herokuapp.com/fonts_list'
    response = requests.get(font_url).text
    font_list = response.split()
    context = {
        'font_list': font_list,
    }
    return render(request, 'artii/artii.html', context)

def artii_result(request):
    string = request.GET.get('string')
    font = request.GET.get('font')
    url = f'http://artii.herokuapp.com/make?text={string}&font={font}'
    res = requests.get(url).text
    context = {
        'artii_result': res,
    }
    return render(request, 'artii/artii_result.html', context)