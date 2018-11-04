
# coding: utf-8

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

