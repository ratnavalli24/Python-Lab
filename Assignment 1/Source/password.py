import re
y = input("enter your password")
z = True
while z:
    if(len(y)<6 and len(y)>16):
        break
    elif not re.search("[a-z]",y) :
        break
    elif not re.search("[A-Z]",y):
        break
    elif not re.search("[0-9]",y):
        break
    elif not re.search("[$#@!]",y):
        break
    else:
        print("valid password")
        z=False
        break
if z:
    print("please enter a valid password")

