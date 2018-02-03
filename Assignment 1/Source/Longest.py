x = input("enter your sentence")
word_list = x.split()
long_word = ' '
for word in word_list:
    if len(word) > len(long_word):
        long_word = word
print(long_word)
