Booklib = {"VLSI":100,"ASE":30,"JAVA":20,"INSTRUMENTATION":40,"MOBILE":35}
for index in Booklib.items():
   print(index)
X=int(input("Enter the minimum value of the range: "))
Y=int(input("Enter the maximum value of the range: "))
S=dict((i, j) for i, j in Booklib.items() if j >= X and j<=Y)
print("You can buy",S.keys())