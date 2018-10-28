
# coding: utf-8

# ### Read data from Console
# To read data from console in Python, use the built-in function input(prompt_string)

# In[2]:


anIntValue=input("Enter an integer value:")
print ("The user has entered this value:", anIntValue)


# ### Print Data to the Console
# 
# To print data to the console, use the in-built function: print(a_string)

# In[3]:


# Examples of using the print()function to print data to the console
print("Examples of using print()function", "\n")
x = 15
print("This is the value of x:",x, "\n")
y = 25
print("This is the value of x:",x, "; This is the value of y:",y, ".\n")


# ### Calculate Diameter and the circumference of a circle. In this scenario, the user inadvertently enters a negative value
# 

# In[5]:


pi=3.14159
radius=float(input("Enter radius: "))
if (radius<0):
    print("This is an error. Enter radius greater than 0")
    radius=float(input("Enter radius: "))
unit=input("Enter units: ") # unit: cms, inches,ft, m
diameter= 2*radius
circumference=diameter * pi
print("Diameter: ",diameter,unit)
print("Circumference: ",circumference,unit)


# ### if Statements

# It is assumed that the registered office of a university asks one analyst to provide a solution to the following problem : Write a python program that can read from the console. The user enters a student's name and his/her level (freshman, ...senior).The program is expected to assign a numeric code that represents his/her priority to register courses. Students who have higher priority are allowed to register courses before those with lower priority. The code starts from 1(highest) that is assigned to seniors and increments by 1 for each lower level.Finally, the program prints out the student name, his/her level, and the code of priority to register courses in the same line

# In[7]:


studentLevel= "Senior" # level: freshman, sophomore, junior, senior
if(studentLevel =="Senior"):
    prioritytoRegister = 1
elif(studentLevel =="Junior"):
    prioritytoRegister = 2
elif(studentLevel =="Sophomore"):
    prioritytoRegister = 3
elif(studentLevel =="Freshman"):
    prioritytoRegister = 4
else:
    print("Invalid studentLevel!!!")
    
print("studentLevel:", studentLevel, "; Priority to register",prioritytoRegister, "\n")   

