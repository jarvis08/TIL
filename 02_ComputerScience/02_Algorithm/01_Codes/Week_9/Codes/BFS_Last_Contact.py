import sys
sys.stdin = open('BFS_Last_Contact.txt', 'r')


for tc in range(1, 11):
    n_node, start = map(int, input().split())
    temp = list(map(int, input().split()))
    graph = dict()
    for i in range(0, int(len(temp)/2)):
        if temp[i*2] not in graph.keys():
            graph[temp[i*2]] = [temp[i*2+1]]
        elif temp[i*2+1] in graph[temp[i*2]]:
            continue
        else:
            graph[temp[i*2]] += [temp[i*2+1]]
    
    queue = [(start, 0)]
    visited = []
    while queue:
        now = queue.pop(0)
        # if now in visited:
        bang = False
        for i in range(len(visited)):
            if visited[i][0] == now[0]:
                bang = True
                break
        if bang:
            continue
        visited.append(now)
        nxt = graph.get(now[0])
        if not nxt:
            continue
        for node in nxt:
            queue.append((node, now[1]+1))
    
    last = visited[-1][1]
    max_last = 0
    for node in visited:
        if node[1] == last:
            if node[0] > max_last:
                max_last = node[0]
    print('#{} {}'.format(tc, max_last))

