import random

ssafy = {
    "location": ["서울", "대전", "구미", "광주"],
    "language": {
        "python": {
            "python standard library": ["os", "random", "webbrowser"],
            "frameworks": {
                "flask": "micro",
                "django": "full-functioning"
            },
            "data_science": ["numpy", "pandas", "scipy", "sklearn"],
            "scraping": ["requests", "bs4"],
        },
        "web" : ["HTML", "CSS"]
    },
    "classes": {
        "seoul":  {
            "lecturer": "john",
            "manager": "jisu",
            "class president": "김병철",
            "groups": {
                "A": ["송치원", "정윤영", "이한얼", "이현빈", "박진홍"],
                "B": ["이수진", "정의진", "임우섭", "김민지", "이건희"],
                "C": ["이여진", "오재석", "김명훈", "이재인", "양찬우"],
                "D": ["김건호", "김윤재", "조동빈", "김병철", "김재현"]
            }
        },
        "gm":  {
            "lecturer": "justin",
            "manager": "pro-gm"
        },
        "gj": {
            "lecturer": "change",
            "manager": "pro-gj"
        }
    }
}


"""
난이도* 1. 지역(location)은 몇개 있나요? : list length
출력예시)
4
"""
print('======== 1 ========')
print(len(ssafy['location']))


"""
난이도** 2. python standard library에 'requests'가 있나요? : 접근 및 list in
출력예시)
False
"""
print('\n======== 2 ========')
print('requests' in ssafy['language']['python']['python standard library'])
print('requests' in ssafy.get('language').get('python').get('python standard library'))


"""
난이도** 3. seoul반의 반장의 이름을 출력하세요. : depth 있는 접근
출력예시)
고승연
"""
print('\n======== 3 ========')
print(ssafy['classes']['seoul']['class president'])
print(ssafy.get('classes').get('seoul').get('class president'))

"""
난이도*** 4. ssafy에서 배우는 언어들을 출력하세요. : dictionary.keys() 반복
출력 예시)
python
web
"""
print('\n======== 4 ========')
for lang in ssafy['language'].keys():
    print(lang)

for lang in ssafy.get("language").keys():
    print(lang)


"""
난이도*** 5 ssafy gm반의 강사와 매니저의 이름을 출력하세요. dictionary.values() 반복
출력 예시)
change
pro-gj
"""
print('\n======== 5 ========')
for name in ssafy.get("classes").get("gm").values():
    print(name)

"""
난이도***** 6. framework들의 이름과 설명을 다음과 같이 출력하세요. : dictionary 반복 및 string interpolation
출력 예시)
flask는 micro이다.
django는 full-functioning이다.
"""
# 방법 1
print('\n======== 6-1 ========')
frame_list = list(ssafy['language']['python']['frameworks'].items())
for frame in frame_list:
    print('{}는 {}이다.'.format(frame[0], frame[1]))

# 방법 2
print('\n======== 6-2 ========')
for k, v in ssafy.get("language").get("python").get("frameworks").items():
    print(f'{k}는 {v}이다.')


"""
난이도***** 7. 오늘 Git pusher 뽑기 위해 groups의 A 그룹에서 한명을 랜덤으로 뽑아주세요. : depth 있는 접근 + list 가지고 와서 random.
출력예시)
오늘의 당번은 하승진
"""
print('\n======== 7 ========')
print(random.choice(ssafy['classes']['seoul']['groups']['A']))
print(random.choice(ssafy.get('classes').get('seoul').get('groups').get('A')))