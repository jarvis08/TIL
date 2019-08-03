compare = [-2, -1, 1, 2]
for case in range(1, 11):
    flex = 0
    num_buildings = int(input())
    buildings = list(map(int, input()[:-1].split(' ')))
    skip = 0
    for i in range(2, num_buildings-2):
        if skip:
            skip -= 1
            continue
        current = buildings[i]
        status = True
        while status:
            for nearby in compare:
                if current <= buildings[i + nearby]:
                    status = False
            if status:
                skip = 2
                flex += 1
                current -= 1
    print('#{} {}'.format(case, flex))