contact_list = [{"name":'Ratnavalli', "number":913, "email":"ratna@gmail.com"},{"name":"Pavan", "number":9134983477, "email":"pavan@gmail.com"},{"name":"Surya","number":816,"email":"surya@gmail.com"}]
nm = input("Enter name to get contact: ")
for i in contact_list:
    if nm in i.values():
        print(i)
num = int(input("Enter number to get contact details: "))
for j in contact_list:
    if num in j.values():
        print(j)
nme = input("Enter name to get contact details and edit number: ")
for k in contact_list:
    if nme in k.values():
        print(k)
        newnum = int(input("Enter number to edit the details of Person: "))
        k["number"] = newnum
        print(k)