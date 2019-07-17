def palindrome(word):
    backward = word[-1::-1]
    if word == backward:
        return True
    return False
print(palindrome('nayana'))