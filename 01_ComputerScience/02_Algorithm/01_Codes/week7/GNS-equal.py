import sys
sys.stdin = open('GNS-Sort_Stringed_Int_input.txt', 'r')

T=int(input())
for t in range(1,T+1):
    input()
    s=input().split()
    l=["ZRO","ONE","TWO","THR","FOR","FIV","SIX","SVN","EGT","NIN"]
    print('#{}'.format(t))
    for n in l:
        for c in s:
            if n==c:
                print(n,end=' ')
    print()