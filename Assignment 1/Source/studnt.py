L1 = ['A','B','C','D','E','F']
L2 = ['B','C','Z','Y','E']
L3 = []
i = 0
for element in L1:
    if element in L2:
        print("common elements are",list(element))
a = 0
for element in L1:
    if element not in L2:
        print("uncommon elements are",list(element))