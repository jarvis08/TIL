def solution(record):
    answer = []
    ids = dict()
    splitted = []
    for i in range(len(record)):
        temp = record[i].split()
        if temp[0] == 'Change':
            ids[temp[1]] = temp[2]
        elif temp[0] == 'Enter':
            ids[temp[1]] = temp[2]
            splitted.append([temp[1], '님이 들어왔습니다.'])
        else:
            splitted.append([temp[1], '님이 나갔습니다.'])
            
    for m in splitted:
        answer.append(ids[m[0]] + m[1])
    return answer

record = ["Enter uid1234 Muzi", "Enter uid4567 Prodo","Leave uid1234","Enter uid1234 Prodo","Change uid4567 Ryan"]
print(solution(record))