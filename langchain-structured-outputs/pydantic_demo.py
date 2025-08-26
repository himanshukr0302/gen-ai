from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'himanshu'
    age: Optional[int] = None
    grade: str = 'A'
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10, default=5, description='A decimal value representing the cgpa of the student')


new_student = {'age': '32', 'email': 'demo@gmail.com'}

student = Student(**new_student) #type:ignore

print(student)