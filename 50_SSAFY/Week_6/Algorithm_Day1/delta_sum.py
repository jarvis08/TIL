l = [[1, 1, 1, 1, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]]
delta_x = [1, 0, -1, 0]
delta_y = [0, 1, 0, -1]
index_range = [0, 1, 2, 3, 4]

summed = 0
for i in range(5):
    for j in range(5):
        for d in range(4):
            new_i = i + delta_x[d]
            new_j = j + delta_y[d]
            if new_i not in index_range or new_j not in index_range:
                continue
            summed += abs(l[i][j] - l[new_i][new_j])

print(summed)