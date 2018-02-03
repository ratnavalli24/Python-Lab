
a=(1,-1,0,2,3,-2)
a = sorted(a)
result = set()
for x in range(len(a)):
    tar = -a[x]
    i,j = x+1, len(a)-1
    while i < j:
        sum = a[i] + a[j]
        if sum < tar:
                i += 1
        elif sum > tar:
                j -= 1
        else:
            result.add((a[x],a[i],a[j]))
            i += 1
            j -= 1
print(result)