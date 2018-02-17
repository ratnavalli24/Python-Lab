class Ratna:
    def __init__(self,name,email):
        self.name = name
        self.email = email
    def display(self):
        print("Name: ", self.name)
        print("Email: ", self.email)
class Student(Ratna):
    StudentCount = 0
    def __init__(self,name,email,student_id):
        Ratna.__init__(self, name, email)
        self.student_id = student_id
        Student.StudentCount +=1
    def displayCount(self):
        print("Total Students:", Student.StudentCount)
    def display(self):
        print("Student Details:")
        Ratna.display(self)
        print("Student Id: ",self.student_id)
class Librarian(Ratna):
    StudentCount = 0
    def __init__(self,name,email,employee_id):
        super().__init__(name,email)
        self.employee_id = employee_id
    def display(self):
        print("Employee Details:")
        Ratna.display(self)
        print("Employee Id: ",self.employee_id)
class Book():
    def __init__(self, bname, author, book_id):
        self.book_name = bname
        self.author = author
        self.book_id = book_id
    def display(self):
        print("Book Details")
        print("Book_Name: ", self.book_name)
        print("Author: ", self.author)
        print("Book_ID: ", self.book_id)
class Borrow_Book(Student,Book):
    def __init__(self, name, email, student_id, bname, author, book_id):
        Student.__init__(self,name,email,student_id)
        Book.__init__(self, bname, author, book_id)
    def display(self):
        print("Borrowed Book Details:")
        Student.display(self)
        Book.display(self)
list1= []
list1.append(Student('Ratnavalli', 'ratna@gmail.com', 913))
list1.append(Student('Pavan', 'pavan@gmail.com', 816))
list1.append(Librarian('Surya', 'surya@gmail.com', 786))
list1.append(Librarian('Sindhu', 'sindhu@gmail.com', 456))
list1.append(Book('Java Programming', 'James Gosling', 399))
list1.append(Book('Python Programming', 'OReily', 209))
list1.append(Borrow_Book('Ratnavalli', 'ratna@gmail.com', 913, 'Java Programming', 'James Gosling', 399))
for obj, item in enumerate(list1):
    item.display()
    print("\n")
    if obj == len(list1)-1:
        item.displayCount()