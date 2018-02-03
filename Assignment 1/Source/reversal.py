x=input("enter a sentence")
word_list=x.split()
for word in word_list:
    print(word[::-1])