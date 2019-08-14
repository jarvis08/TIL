word = 'Python'
l_word = len(word)
f_word = ''
for i in range(l_word-1, -1, -1):
    f_word += word[i]
print(f_word)

s_word = word[::-1]
print(s_word)

r_word = reversed(word)
print(''.join(r_word))