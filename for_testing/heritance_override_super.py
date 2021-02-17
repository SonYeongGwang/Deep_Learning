'''
inheritance
'''
# class introduce:

#     def say_hello(self):
#         print('hello, ')

# class occupation(introduce):
#     def __init__(self,name , occu):
#         self.occu = occu
#         self.name = name
    
#     def say_occu(self):
#         print('My name is {}, Im {}.'.format(self.name, self.occu))

# son = occupation('Son', 'Graduate Student')
# son.say_hello()
# son.say_occu()


'''
override
'''
# class Country:
#     """Super Class"""

#     name = '국가명'
#     population = '인구'
#     capital = '수도'

#     def show(self):
#         print('국가 클래스의 메소드입니다.')

# class Korea(Country):
#     """Sub Class"""

#     def __init__(self, name,population, capital):
#         self.name = name
#         self.population = population
#         self.capital = capital

#     def show(self):
#         super().show()
#         print(
#             """
#             국가의 이름은 {} 입니다.
#             국가의 인구는 {} 입니다.
#             국가의 수도는 {} 입니다.
#             """.format(self.name, self.population, self.capital)
#         )

# Kor = Korea("대한민국", "5,182만1669명", "서울")
# Kor.show()

class Parent:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        
class Child(Parent):
    def __init__(self, c1, **kwargs):
        super(Child, self).__init__(**kwargs) # super()를 사용하지 않으면 Parent의 __init__가 overriding 됩니다.
        self.c1 = c1
        self.c2 = "This is Child's c2"
        self.c3 = "This is Child's c3"

child = Child(p1="This is Parent's p1", 
	      p2="This is Parent's p1", 
              c1="This is Child's c1")

print(child.p1)
print(child.p2)
print(child.c1)
print(child.c2)
print(child.c3)