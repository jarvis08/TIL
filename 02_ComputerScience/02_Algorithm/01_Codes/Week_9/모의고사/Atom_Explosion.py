import sys
sys.stdin = open('Atom_Explosion.txt', 'r')


TC = int(input())
for tc in range(1, TC+1):
    n_atom = int(input())
    atoms = []
    for i in range(n_atom):
        atoms.extend([list(map(int, input().split()))])

    # 상 하 좌 우
    # 0 1 2 3
    dx = [0, 0, -0.5, 0.5]
    dy = [0.5, -0.5, 0, 0]

    explosion = 0
    k = 0
    while k < 4000:
        # 이동
        for atom in atoms:
            atom[0] = atom[0] + dx[atom[2]]
            atom[1] = atom[1] + dy[atom[2]]

        # 같은 좌표 찾기
        boms = []
        for i in range(n_atom):
            if i in boms:
                continue
            boom = False
            standard = atoms[i][:2]
            for j in range(n_atom):
                if i == j:
                    continue
                if j in boms:
                    continue
                if standard == atoms[j][:2]:
                    boom = True
                    explosion += atoms[j][3]
                    boms.append(j)
            if boom:
                explosion += atoms[i][3]
                boms.append(i)
        if boms:
            for idx in reversed(sorted(boms)):
                atoms.pop(idx)
                n_atom -= 1
        if len(atoms) < 2:
            break
        k += 1
    print(explosion)