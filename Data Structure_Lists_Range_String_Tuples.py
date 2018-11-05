
# coding: utf-8

# # Data Structure_Lists

# ### 1.2 Properties of Lists

# In[2]:


# empty list
my_lists=[]


# In[3]:


# list of integers
my_list=[1,2,3]


# In[4]:


#list with mixed datatypes
my_list=[1, "Hello",3.4]


# In[5]:


# nested list
my_list=["mouse",[8,4,6],['a']]


# ### 1.4 Constructor list (iterable)

# In[6]:


list ("abc")


# In[7]:


list ((1,2,3))


# In[8]:


list([1,3,5,7,9])


# ### 2. Create Lists

# In[9]:


empty_list = []
another_empty_list=list ()
print(len(empty_list))
print(len(another_empty_list))


# In[11]:


### 2.3. Create lists by converting other data structures/types to lists: Using list()


# ### 2.3.1 Convert list from strings or tuples using the constructor list ()

# In[12]:


# Convert a string of one word  to list of characters
list("house")


# In[13]:


# convert a string  of words to a list of characters
list("This word")


# In[14]:


#Convert a tuple to a list
aTuple =('ready','fire','aim')
list(aTuple)


# ### 2.3.2 Create lists from strings using split() method

# In[15]:


# Convert a string of words to a list of words :Using split() to chop the string with ' ' as
aStringOfWords= "This is a string of words"
aList=aStringOfWords.split(' ')
print(aList)


# In[16]:


# Convert a string to a list: Using split() to chop the string with some separator
aDayString="5/1/2017"
aList=aDayString.split('/')
print(aList)


# ### 2.3.3. Create lists by using comprehension and slicing a existing list

# In[17]:


l_lists=[[1,2,3],[2,3,4],[3,4,5]]
#Notes: Must use list slice--> cannot use any other function to delete/remove
new_llists=[element[1:] for element in l_lists]
i=0
for element in new_llists:
    print(element)
    i=i+1
    if i==3:
        break


# ## 3. Access List Elements
# ### 3.1 Access single elements

# In[18]:


my_list=['p','r','o','b','e']
# output:p
print(my_list[0])
# output:o
print(my_list[2])
# output:e
print(my_list[4])
# nested list
n_list=["Happy",[2,0,1,5]]
# Nested Indexing
#Output:a
print(n_list[0][1])
#Output:5
print(n_list[1][3])


# In[19]:


# Access using forward Index
aTuple=('ready','fire','aim')
aList=list(aTuple)
print("Length of the list:",len(aList))
list_element1=aList[0]
list_element2=aList[1]
list_element3=aList[2]
print(list_element1)
print(list_element2)
print(list_element3)


# In[20]:


languages=["Python","C","C++","Java","Perl"]
print(languages[0]+" and "+languages[1]+"are quite different !")


# ## 3.2 Access a slice of lists

# In[21]:


# we can access a range of items in a list by using the slicing operator(colon)
my_list=['p','r','o','g','r','a','m','i','z']

# elements 3rd to 5th
print(my_list[2:5])

# elements beginning to 4th
print(my_list[:-5])

# elements beginning to 4th
print(my_list[5:])

# elements beginning to end
print(my_list[:])


# ## 4. Modify Lists
# ### 4.1 Add/change elements of lists

# In[22]:


# 4.1.1 Update/Change single elements or a sub-list of lists
odd= [2,4,6,8]

# change the 1st item
odd[0]=1

# Output:[1,4,6,8]
print(odd)

# change 2nd to 4th items
odd[1:4]=[3,5,7]

# Output:[1,3,5,7]
print(odd)


# In[23]:


# 4.1.2 Add single items or a sub-list into a list- using append() or extend() respectively
# We can add one item to a list using append() method
# or add several items using extend() method
odd= [1,3,5]

odd.append(7)

# Output:[1,3,5,7]
print(odd)

odd.extend([9,11,13])

# Output :[1,3,5,7,9,11,13]
print(odd)


# In[24]:


# 4.1.3 Insert single elements or sub-lists into an existing list
# We can insert one item to a desired location by using the method insert()
# or insert several multiple items by squeezing it into an empty slice of a list

odd=[1,9]
odd.insert(1,3)

# Output :[1,3,9]
print(odd)

odd[2:2]=[5,7]

# Output:[1,3,5,7,9]
print(odd)


# ## 4.2 Delete/Remove elements of lists
# 
# ### 4.2.1 Delete/Remove elements of lists-using the del() function

# In[25]:


# We can delete one or more items from a list using the keyword del.

my_list=['p','r','o','b','l','e','m']

# delete one item
del my_list[2]

# Output: ['p','r',b','l','e','m']
print("3rd element has been removed: ",my_list)

# delete multiple items
del my_list[1:5]

# Output: ['p','m']
print("Elements from index 1 until 4 have been removed: ",my_list)


# ### 4.2.2 Delete/Remove elements of lists-using the functions remove() or pop()
# 
# 

# In[26]:


# We can use remove() method to remove the given item or pop() method to remove an item 
# The pop()method removes and returns the last item if index is not provided
# This helps us implement lists as stacks(first in,last out data structure)
# We can also use the clear() methodto empty a list

my_list=['p','r','o','b','l','e','m']
my_list.remove('p')

#output:['r','o','b','l','e','m']
print(my_list)

#output:'o'
print(my_list.pop(1))

#output:['r','b','l','e','m']
print(my_list)

#output:'m'
print(my_list.pop())

#output:['r','b','l','e']
print(my_list)


# ### 4.2.3 Delete/Remove elements of a list-assigning an empty list [ ] to a slice of the list

# In[27]:


my_list=['p','r','o','b','l','e','m']

# Remove o
my_list[2:3]=[]

print(my_list)

# Remove b,l,e
my_list[2:5]=[]

print(my_list)


# ### 4.2.3 Delete/Remove all the elements of a list- using the clear() function

# In[28]:


my_list=['p','r','o','b','l','e','m']

my_list.clear()

#Output: []
print(my_list)


# ## 5. Copy Lists
# ### 5.1 Shallow Copy - Only reference  to the object is copied. No new object is created

# In[29]:


# Intializing list 1
list1=[1,2,[3,5],4]

# using copy
list2 = list1

# The lists refer to the same objects, i.e., same id values
id(list1), id(list2)


# ### 5.2 Deep Copy - A new object will be created when the copying has done

# In[30]:


# importing "copy" for copy operations
import copy

# initializing list 1
list1=[1,2,[3,5],4]

# using deepcopy to deepcopy
list2=copy.deepcopy(list1)

#The lists refer to different objects, i.e., different id values
id(list1), id(list2)


# ## 6. Delete Lists

# In[31]:


list1=[1,2,[3,5],4]
print(list1)
del(list1)
print("list1 has been deleted.")


# In[32]:


print(list1)


# ### 6.1 Concatenante lists

# In[34]:


# Using + to concatenate strings
list1=[1,2,[3,5],4]
list2=["Hello","World"]
print(list1 + list2)


# In[35]:


# We can also use + operator to combine 2 lists. This is also called concatenantion.
# The * operator repeats a list for the given number of times.

odd=[1,3,5]

# Output:[1,3,5,9,7,5]

print(odd+[9,7,5])


# ### 6.2 Replicate Lists

# In[36]:


aList=[1,2]
print(aList*3)


# In[37]:


# Output:["re","re","re"]
print(["re"]*3)


# ### 6.3 Test elements with "in" and "not in"
# 
# 

# In[38]:


list1=[1,2,[3,5],4]
print(2 in list1)
print([3] in list1)


#  ### 6.4 Compare lists: <,>,<=,>=,==,!=

# In[39]:


list1=[1,2,[3,5],4]
list2=[1,2,4]
print(list1==list2)


# ### Iterate a list using for loop

# In[40]:


list1=[1,2,[3,5],4]

for i in list1:
    print(i)


# In[41]:


list1=[1,2,[3,5],4]

for i in list1:
    print(i,end="")


# In[42]:


list1=[1,2,[3,5],4]

for i in list1:
    print(i,end="\n")


# In[43]:


for fruit in['apple','banana','mango']:
    print("I like",fruit)


# ### 6.6 Sort Lists

# ### 6.6.1 Using the sort method of the class list:sort(*,key=none,reverse=false)

# In[45]:


# vowels list
vowels=['e','a','u','o','i']

# sort the vowels
vowels.sort()

# Print vowels
print('sorted list:',vowels)


# ### 6.6.2 Using the built-in sorted() function:sorted(iterable,*,key=None,reverse=False)

# In[46]:


# the built-in sorted()function returns a new sorted list from the items in iterable.

# vowels list
vowels=['e','a','u','o','i']

# sort the vowels
sortedVowels=sorted(vowels)

# Print vowels
print('Sorted list:',sortedVowels)

# A new list has been created and returned by the built-in sorted function

id(vowels), id(sortedVowels)


# ## 7. Class list
# 
# ### 7.1 Helpful methods
# 
# 

# In[47]:


# 7.1.1. count()- count(x) returns the number of elements of the tuple that are equal to x


# In[49]:


list1=['a','p','p','l','e']

# Count
# Output: 2
print(list1.count('p'))


# In[50]:


### 7.1.2 index(x) - returns the index of the first element that is equal to x

list1=['a','p','p','l','e']

# Count
# Output: 2
print(list1.index('p'))


# # Data Structure -Range

# ## 2.1 Using range() in creating other sequence objects

# In[52]:


list(range(10))


# In[53]:


list(range(1,11))


# In[54]:


list(range(0,30,5))


# In[55]:


list(range(0,10,3))


# In[56]:


list(range(0,-10,-1))


# In[57]:


list(range(0))


# list(range(1,0))

# # Data Structure- Strings

# In[59]:


aStr="Hello"
print(aStr)

aStr2='Hello'
print(aStr2)


# ### 1.3 Length of Strings

# In[62]:


# Declare a string
aStr="This is a String."
print("The length of this string -or the number of characters: ", len(aStr))


# ### 1.3 String Indices

# In[64]:


aStr="This is a String."

print("The length of this string -or the number of characters: ", len(aStr))
print(aStr[0])
print(aStr[1])
print(aStr[16])


# In[65]:


print(aStr[17])


# ## 2. Create Strings
# ### 2.1. Using String Literals

# In[67]:


# all the following are equivalent
my_string='Hello'
print(my_string)

my_string="Hello"
print(my_string)

my_string='''Hello'''
print(my_string)

#triple quotes strings can extend multiple lines
my_string= """Hello.Welcome to 
             Python World!"""
print(my_string)


# ### 2.2 Create Strings from Lists-Using join() method

# In[68]:


# VERSION 1: List of strings -> A string

aList=["This","is","a","string"]

print("This is a list: ", aList)

aString = " ".join(aList)

# aString is a string:"This is a string"
print(aString)


# ### 2.3 Create Strings from Lists - Using str() and join()

# In[69]:


# VERSION 2: List of numbers -> A string

# A List of numbers
aList = [20,30,40,50,60]

# Convert aList into a list of strings-Using the constructor str()
aStrList = [str(element) for element in aList]
print("This is a list of strings: ",aStrList)


# Using join() to create a new string
aString=" ".join(aStrList)

# aString= ['20','30','40','50','60']
print("This is a string: ",aString)


# ### 2.4 Create Strings from Lists- Using map() and join()

# In[70]:


# Generate the combination from the list
# Then transform each element of the list into a string

from itertools import combinations
L=[1,2,3,4]
print(combinations(L,3))

# Using map() and join() to convert each numeric combination into a string
# Thanks to the tecnhique, we can display he list of combinations
[",".join(map(str,comb)) for comb in combinations(L,3)]


# ## 3. Access Characters in Strings

# ### 3.1 Access Single Characters

# In[71]:


# Python allows negative indexing for its sequences.
# The index of -1 refers to the last item, -2 to the second last item and so on.
# We can access a range of items in a string by using the slicing operator(colon).

str='programiz'
print('str= ', str)

# first character
print('str[0]',str[0])

# Third character
print('str[0]',str[2])

# Last character
print('str[-1]',str[-1])


# ### 3.2 Access a Slice of Strings

# In[72]:


str='programiz'
# slicing 2nd to 5th character
print('str[1:5]= ', str[1:5])

# slicing 6th to 2nd last character
print('str[5:-2]= ', str[5:-2])

sample_str='Python String'
print(sample_str[3:5]) # Print a range of character starting from index 3 to index 4

#ho
print(sample_str[7:]) # Print all characters starting from index 7

#string
print(sample_str[:6]) # Print all characters before index 6

#Python
print(sample_str[7:-4]) # Print all characters from index 7 to the index -4


# ## 4. Modify Strings 

# In[73]:


# strings are immutable.Any attempt to change or modify contents will lead to errors.

sample_str='Python String'
sample_str[2]='a'


# In[74]:


# But an existing string variable can be reassigned with a brand new string

str1 ="This is a string."
print("str1: ",str1 )

# Reassign a new tuple to tuple1
str1= "This is a new string."
print("str1 after being reassigned: ",str1 )


# ## 5. Copy Strings
# 
# ### 5.1 Shallow Copy

# In[75]:


str1= "Hello"
str2=str1

# Both the strings refer to the same object, i.e, the same id value

id(str1), id(str2)


# ## 6. Delete Strings

# In[76]:


sample_str= "Python is the best scripting language"
del(sample_str )


# In[77]:


# to show that the string has been deleted, let's print it

# => error
print(sample_str)


# ## 7.Operations on Strings

# ### 7.1 Concatenate Strings

# In[80]:


str1='Hello'
str2=' '
str3='World!'
# using+
print('str1+str2+str3= ',str1+str2+str3)


# ### 7.2 Replicate Strings

# In[81]:


# Using * to replicate  string

str="Hello"
replicatedStr=str*3
print("the string has been replicated three times: ", replicatedStr)


# ### 7.3 Test substrings with "in" & "not in" 

# In[82]:


str1="Welcome"
print("come" in str1)
print("come" not in str1)


# ## 7.4 Compare strings: <, >, <=,==,!=

# In[83]:


# TRUE:"apple" comes before "banana"
print("apple" < "banana")
print("apple" < "Apple")

print("apple"=="Apple")


# ### 7.5 Iterate strings using for loops

# In[84]:


aStr="Hello"

for i in aStr:
    print(i)


# In[85]:


aStr="Hello"

for i in aStr:
    print(i,end="")


# In[86]:


aStr="Hello"

for i in aStr:
    print(i,end="\n")


# ### 7.6 Test Strings

# In[87]:


s = "welcome to Python"
s.isalnum()


# In[88]:


"Welcome".isalpha()


# In[89]:


"first Number".isidentifier()


# In[90]:


"WELCOME".isupper()


# In[91]:


"Welcome".islower()


# In[93]:


"s".islower()


# In[94]:


" \t".isspace()


# ## 8. Class String

# ### 8.1 Helpful methods
# 
# 8.1.1 count (x)- returns the number of elements of the tuple that are equal to x

# In[96]:


str1= "This is a string: Hello....Hello Python World!"
print(str1.count("Hello"))


# 8.1.2 index(x) - returns the index of the first element that is equal to x

# In[97]:


str1= "This is a string: Hello....Hello Python World!"
print(str1.index('s'))


# # Data Structure Tuples

# In[98]:


t= ("tuples","are","immutable")
t[0]


# In[99]:


t=("tuples","are","immutable")

print(t[0])
print(t[-1])
print(t[-3])

print(len(t))


# ### 2. Create Tuples

# In[101]:


# empty tuple
# Output: ()
my_tuple=()
print(my_tuple)


# In[102]:


# tuple having integers
# Output: (1,2,3)
my_tuple=(1,2,3)
print(my_tuple )


# In[103]:


# tuple with mixed datatypes
# Output: (1, "Hello",3.4)
my_tuple= (1,"Hello", 3.4)
print(my_tuple)


# In[104]:


# nested tuple
# output:("mouse",[8,4,6],(1,2,3))
my_tuple=("mouse",[8,4,6],(1,2,3))
print(my_tuple)


# ### 2.2 Create tuples with only one elements

# In[106]:


# only paratheses is not enough

# Output: <class 'str'>
my_tuple=("hello")
print(type(my_tuple))


# In[108]:


# need a comma at the end
# Output :<class 'str'>
my_tuple=("hello",)
print(type(my_tuple))


# In[109]:


# paranthesis is optional
# output :<class 'tuple'>
my_tuple="hello",
print(type(my_tuple))


# ## 3.  Access List Elements
# ### 3.1 Access single elements of Tuples

# In[112]:


my_tuple=('p','e','r','m','i','t')
# output :'p'
print(my_tuple[0])

# output :'t'
print(my_tuple[5])

# nested tuple
n_tuple=("mouse",[8,4,6],(1,2,3))

# nested index
# output:'s'
print(n_tuple[0][3])

# nested index
# output:4
print(n_tuple[1][1])


# In[113]:


my_tuple=('p','e','r','m','i','t')
# output :'t'
print(my_tuple[-1])

# output :'p'
print(my_tuple[-6])


# In[114]:


my_tuple=('p','e','r','m','i','t')
# Range of the indices:0..len(my_tuples)-1:0..6

# index must be in range
# Error: out of range
print(my_tuple[6])


# ### 3.2 Access a slice of tuples

# In[115]:


my_tuple =('p','r','o','g','r','a','m','i','z')

# elements 2nd to 4th
# Output:('r','o','g')
print(my_tuple[1:4])

# elements beginning to 2nd
# Output:('p','r')
print(my_tuple[:-7])

# elements 8th to end
# Output:('i','z')
print(my_tuple[7:])

# elements beginning to end
# Output:('p','r','o','g','r','a','m','i','z')
print(my_tuple[:])


# ## 4. Modify Tuples

# In[116]:


#4.1 Tuples are immutable.They cannot be changed after being created

aTuple =('Python','C','C++','Java','Scala')
aTuple[2]='Ruby'


# In[122]:


# 4.2 Tuples:One or more elements are mutable objects:lists,bytearrays,etc.

my_tuple=(4,2,3,[6,5])

# but item of mutable element can be changed
# Output: (4,2,3,[9,5])

my_tuple[3][0]=9
print(my_tuple)


# ## 4.3 Tuples:Tuples can be reassigned
# 
# 

# tuple1=(4,2,3,[6,5])
# print("tuple1: ",tuple1)
# 
# # Reassign a new tuple to tuple1
# 
# my_tuple =('p','r','o','g','r','a','m','i','z')
# tuple1=my_tuple
# print("tuple1 after being reassigned: ", tuple1)
# 

# ## 5. Copy Tuples
# 
# ### 5.1 Shallow Copy

# In[124]:


tuple1="Hello"
tuple2=tuple1

# Both the tuples refer to the same object, i.e, the same id value
id(tuple1),id(tuple2)


# ### Delete Tuples
# 
# 

# In[126]:


aTuple="Python is the best scripting language"

del(aTuple)


# In[127]:


# to show that the string has been deleted, let's print it

# -> error

print(aTuple)


# ## 6. Operations on Tuples
# 
# ### 6.1 Concatenate Tuples

# In[128]:


tuple1='Hello' # comma to indicate this is a tuple; paranthesis are optional
tuple2=' '
tuple3='World!'
# using+
print('tuple1+tuple2+tuple3= ',tuple1+tuple2+tuple3)


# ### 6.2 Replicate Tuples

# In[131]:


# Using * to replicate a tuple

tuple1="Hello",
replicatedTuple=tuple1*3
print(replicatedTuple)


# ### 6.3Test substrings with "in" & "not in"
# 

# In[132]:


aTuple=(2,4,6,"This","is","a","tuple")
print(2 in aTuple)
print('a' in aTuple)
print("This is" in aTuple)


# ### 6.4 Compare tuples: <,>,<=,>=,==,!=
# 

# In[133]:


tuple1="Hello World!"
tuple2="hello world!" 
print(tuple1==tuple2)


# ### 6.5 Iterate a tuple using for loop
# 
# 

# In[135]:


tuple1=("This","is",1,"book")

for i in tuple1:
    print(i)


# In[136]:


tuple1=("This","is",1,"book")

for i in tuple1:
    print(i,end="")


# In[137]:


tuple1=("This","is",1,"book")

for i in tuple1:
    print(i,end="\n")


# ## 7. Class tuple

# ### 7.1.1.count()

# In[139]:


my_tuple=('a','p','p','l','e',)

# Count
# Output:2
print(my_tuple.count('p'))


# ### 7.1.2.index(x)

# In[141]:


# Index
# Output:3
print(my_tuple.index('l'))

