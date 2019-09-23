lunch = {
    '진가와' : '01011112222',
    '대우식당' : '01054813518',
    '바스버거' : '01088465846'
}
# 1. lunch.csv 데이터 저장
with open('lunch.csv', 'w', encoding='UTF-8') as f:
    for k, v in lunch.items():
        f.write(f'{k},{v}\n')

# 2. ',' join을 사용하여 string 만들기
with open('lunch.csv', 'w', encoding='UTF-8') as f:
    for item in lunch.items():
        f.write(','.join(item))

# 3. csv 라이브러리 사용
# writer와 reader가 따로 존재
import csv
with open('lunch.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    for item in lunch.items():
        csv_writer.writerow(item)

# 4. csv.DictWriter()
# field name을 설정 가능
with open('student.csv', 'w', encoding='utf-8', newline='') as f:
    fieldnames = ['name', 'major']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({'name':'john', 'major':'cs'})
    writer.writerow({'name':'dongbin', 'major':'ie'})