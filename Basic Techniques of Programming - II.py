
# coding: utf-8

# It is assumed that a software developer is asked to write a Python program that can calculate and print out the diameter and circumference of a circle. The user enters data of the radius and its measurement unit(in, ft,cms or m) from the console

# In[ ]:


pi=3.14159
radius=float(input("Enter radius: "))
if (radius<0):
    print("This is an error. Enter radius greater than 0")
    radius=float(input("Enter radius: "))
unit=input("Enter units: ") # unit: cms,inches,ft,m
if(unit is not among (inches,ft,cms,m):
   unit=input("Enter the correct unit from the list: ")
   
diameter= 2*radius
circumference=diameter * pi
print("Diameter: ",diameter,unit)
print("Circumference: ",circumference,unit)


# ### Calculating the diameter and circumference of a circle if a user makes mistakes while entering the radius again and again
# 
# The Program must repeatedly check the input until it gets the correct one-> The program uses LOOPS

# In[1]:


# while LOOP
pi=3.14159
radius=float(input("Enter radius: "))
while(radius<0):
    print("This is an error. Enter radius greater than 0")
    radius=float(input("Enter radius: "))
unit=input("Enter units: ") # unit: cms,inches,ft,m
diameter= 2*radius
circumference=diameter * pi
print("Diameter: ",diameter,unit)
print("Circumference: ",circumference,unit)


# In[2]:


# if-else
numCredits = int(input("Enter # of credits then press Enter:"))

if(numCredits >= 120):
    readyToGraduate = True;
else:
    readyToGraduate = False;

print(readyToGraduate)


# In[3]:


# for Loop
language=["C","C++","Java","Python","Perl","Ruby","Scala"]

for x in language:
    print(x)


# ### Break Statement 
# Loop will be exited. The program flow will continue following the FOR LOOP, if there is any at all

# In[4]:


edibles=["ham","spam","eggs","nuts"]
for food in edibles:
    if food== "spam":
        print("No More spam please!")
        break
    print("Great,delicious " + food)    
else:
          print("I'm so glad; No spam!")
print("Finally, I finished stuffing myself")        
          
          
          


# In[6]:


#when no break,FOR loop iterated through the whole sequence.The output is misleading since spam is in sequence
edibles=["ham","spam","eggs","nuts"]
for food in edibles:
    print("No break in FOR loop statements")
else:
    print("I'm so glad; No spam!")
    


# In[9]:


# when no else 
edibles=["ham","spam","eggs","nuts"]
spam=False

for food in edibles:
    if food== "spam":
        spam=True
        print("No More spam please!")
        break
    print("Great,delicious " + food) 
    
if(not spam):
    print("I'm so glad; No spam!")
    
print("Finally, I finished stuffing myself")


# In[10]:


# no spam
edibles=["ham","eggs","nuts"]
spam=False

for food in edibles:
    if food== "spam":
        spam= True
        print("No More spam please!")
        break
    print("Great,delicious " + food)  
    
if(not spam):
    print("I'm so glad; No spam!")
    
print("Finally, I finished stuffing myself")  


# In[12]:


# Break is to terminate a loop - completely get out of the loop, immediately
numItems=0
totalSales=0
totalSoldItems=0

while(numItems<totalSoldItems):
    price=int(input("Enter the price of the next sold item: "))
    totalSales=totalSales+price
    if(totalSales>=1000000):
        break;
    numItems=numItems+1 
print(totalSales)  


# In[1]:


edibles=["ham","spam","eggs","nuts"]
spam=False

for food in edibles:
    if food== "spam":
        spam=True
        print("No More spam please!")
        continue
    print("Great,delicious " + food) 
    
if(not spam):
    print("I'm so glad; No spam!")
    
print("Finally, I finished stuffing myself")  

