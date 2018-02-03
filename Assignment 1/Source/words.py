x = input("enter your sentence")
word_list = x.split()
y=len(word_list)
if(y%2==0):
    first = int(y/2)
    sec = int(y/2)-1
    print("the length of the sentence is",len(word_list))
    print("the middle words are",word_list[sec],"and",word_list[first])
else:
     z=int((y/2))
     print("the length of the sentence is",len(word_list))
     print("the middle word is",word_list[z])
