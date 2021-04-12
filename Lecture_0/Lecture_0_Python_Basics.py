##COPYRIGHT from 仲益教育

#1 HELLO WORLD!
# Python is a very simple language, and has a very straightforward syntax. 
# It encourages programmers to program without boilerplate (prepared) code. 
# The simplest directive in Python is the "print" directive - it simply prints out a line 
# (and also includes a newline, unlike in C).
# To print a string in Python 3, just write:
print("This line will be printed.")

#1.1 Indentation
# Python uses indentation for blocks, instead of curly braces. Both tabs and spaces are supported, 
# but the standard indentation requires standard Python code to use four spaces. For example:
x = 1
if x == 1:
    # indented four spaces
    print("x is 1.")

#2 Variables and Types
# Python is completely object oriented, and not "statically typed". 
# You do not need to declare variables before using them, or declare their type. 
# Every variable in Python is an object.
# This tutorial will go over a few basic types of variables.

#2.1 Numbers
# Python supports two types of numbers - integers and floating point numbers. 
# (It also supports complex numbers, which will not be explained in this tutorial).
# To define an integer, use the following syntax:
myint = 7
print(myint)
#To define a floating point number, you may use one of the following notations:
myfloat = 7.0
print(myfloat)
myfloat = float(7)
print(myfloat)

#2.2 Strings
#Strings are defined either with a single quote or a double quotes.
mystring = 'hello'
print(mystring)
mystring = "hello"
print(mystring)
#The difference between the two is that using double quotes makes it easy to include apostrophes 
#(whereas these would terminate the string if using single quotes)
mystring = "Don't worry about apostrophes"
print(mystring)

#There are additional variations on defining strings that make it easier to include things such as carriage returns, 
#backslashes and Unicode characters. These are beyond the scope of this tutorial, 
#but are covered in the Python documentation.
#Simple operators can be executed on numbers and strings:
one = 1
two = 2
three = one + two
print(three)

hello = "hello"
world = "world"
helloworld = hello + " " + world
print(helloworld)

#Assignments can be done on more than one variable "simultaneously" on the same line like this
a, b = 3, 4
print(a,b)
# Mixing operators between numbers and strings is not supported:
# This will not work!
one = 1
two = 2
hello = "hello"
print(one + two + hello)

#3 Lists
# Lists are very similar to arrays. They can contain any type of variable, 
# and they can contain as many variables as you wish. Lists can also be iterated over in a very simple manner. 
# Here is an example of how to build a list.
mylist = []
mylist.append(1)
mylist.append(2)
mylist.append(3)
print(mylist[0]) # prints 1
print(mylist[1]) # prints 2
print(mylist[2]) # prints 3

# prints out 1,2,3
for x in mylist:
    print(x)
    
#Accessing an index which does not exist generates an exception (an error).
mylist = [1,2,3]
print(mylist[10])

#4 Basic Operators
#4.1 Arithmetic Operators
#Just as any other programming languages, the addition, subtraction, multiplication, 
#and division operators can be used with numbers.
number = 1 + 2 * 3 / 4.0
print(number)

# Try to predict what the answer will be. Does python follow order of operations?
# Another operator available is the modulo (%) operator, 
# which returns the integer remainder of the division. dividend % divisor = remainder.
remainder = 11 % 3
print(remainder)

#Using two multiplication symbols makes a power relationship.
squared = 7 ** 2
cubed = 2 ** 3
print(squared)
print(cubed)

#4.2 Using Operators with Strings
#Python supports concatenating strings using the addition operator:
helloworld = "hello" + " " + "world"
print(helloworld)

#Python also supports multiplying strings to form a string with a repeating sequence:
lotsofhellos = "hello" * 10
print(lotsofhellos)

#4.3 Using Operators with Lists
#Lists can be joined with the addition operators:
even_numbers = [2,4,6,8]
odd_numbers = [1,3,5,7]
all_numbers = odd_numbers + even_numbers
print(all_numbers)
#Just as in strings, Python supports forming new lists with a repeating sequence using the multiplication operator:
print([1,2,3] * 3)

#5 String Formatting
# Python uses C-style string formatting to create new, formatted strings. 
# The "%" operator is used to format a set of variables enclosed in a "tuple" (a fixed size list), 
# together with a format string, which contains normal text together with "argument specifiers", 
# special symbols like "%s" and "%d".

# Let's say you have a variable called "name" with your user name in it, 
# and you would then like to print(out a greeting to that user.)
# This prints out "Hello, John!"
name = "John"
print("Hello, %s!" % name)
# To use two or more argument specifiers, use a tuple (parentheses):
# This prints out "John is 23 years old."
name = "John"
age = 23
print("%s is %d years old." % (name, age))
# Any object which is not a string can be formatted using the %s operator as well. 
# The string which returns from the "repr" method of that object is formatted as the string. For example:
# This prints out: A list: [1, 2, 3]
mylist = [1,2,3]
print("A list: %s" % mylist)

#6 Conditions
# Python uses boolean variables to evaluate conditions. 
# The boolean values True and False are returned when an expression is compared or evaluated. 
# For example:
x = 2
print(x == 2) # prints out True
print(x == 3) # prints out False
print(x < 3) # prints out True

# Notice that variable assignment is done using a single equals operator "=", 
# whereas comparison between two variables is done using the double equals operator "==". 
# The "not equals" operator is marked as "!=".

# 6.1 Boolean operators
# The "and" and "or" boolean operators allow building complex boolean expressions, for example:
name = "John"
age = 23
if name == "John" and age == 23:
    print("Your name is John, and you are also 23 years old.")

if name == "John" or name == "Rick":
    print("Your name is either John or Rick.")
    
# 6.2 The "in" operator
# The "in" operator could be used to check if a specified object exists within an iterable object container, 
# such as a list:
name = "John"
if name in ["John", "Rick"]:
    print("Your name is either John or Rick.")

# 6.3 If operator
# Python uses indentation to define code blocks, instead of brackets. 
# The standard Python indentation is 4 spaces, although tabs and any other space size will work, 
# as long as it is consistent. Notice that code blocks do not need any termination.
# Here is an example for using Python's "if" statement using code blocks:
statement = False
another_statement = True
if statement is True:
    # do something
    pass
elif another_statement is True: # else if
    # do something else
    pass
else:
    # do another thing
    pass

#another example
x = 2
if x == 2:
    print("x equals two!")
else:
    print("x does not equal to two.")
    

    
# A statement is evaulated as true if one of the following is correct: 1. 
# The "True" boolean variable is given, or calculated using an expression, 
# such as an arithmetic comparison. 2. An object which is not considered "empty" is passed.
# Here are some examples for objects which are considered as empty: 1. An empty string: "" 
# 2. An empty list: [] 3. The number zero: 0 4. The false boolean variable: False

# 6.4 The 'is' operator
# Unlike the double equals operator "==", the "is" operator does not match the values of the variables,
#  but the instances themselves. For example:
x = [1,2,3]
y = [1,2,3]
print(x == y) # Prints out True
print(x is y) # Prints out False

# 6.5 The "not" operator
# Using "not" before a boolean expression inverts it:
print(not False) # Prints out True
print((not False) == (False)) # Prints out False

#7 Loops
# There are two types of loops in Python, for and while.

# 7.1 The "for" loop
# For loops iterate over a given sequence. Here is an example:
primes = [2, 3, 5, 7]
for prime in primes:
    print(prime)

# For loops can iterate over a sequence of numbers using the "range" and "xrange" functions. 
# difference between range and xrange is that the range function returns a new list 
# with numbers of that specified range, 
# whereas xrange returns an iterator, which is more efficient. 
# (Python 3 uses the range function, which acts like xrange). 
# Note that the range function is zero based.

# Prints out the numbers 0,1,2,3,4
for x in range(5):
    print(x)

# Prints out 3,4,5
for x in range(3, 6):
    print(x)

# Prints out 3,5,7
for x in range(3, 8, 2):
    print(x)

# 7.2 While Loops
#While loops repeat as long as a certain boolean condition is met. For example:
count = 0
while count < 5:
    print(count)
    count += 1  # This is the same as count = count + 1

# 7.3 "break" and "continue" statements

# break is used to exit a for loop or a while loop,
# whereas continue is used to skip the current block, 
# and return to the "for" or "while" statement. A few examples:
    
# Prints out 0,1,2,3,4
count = 0
while True:
    print(count)
    count += 1
    if count >= 5:
        break

# Prints out only odd numbers - 1,3,5,7,9
for x in range(10):
    # Check if x is even
    if x % 2 == 0:
        continue
    print(x)

# can we use "else" clause for loops?
# unlike languages like C,CPP.. we can use else for loops. 
# When the loop condition of "for" or "while" statement fails then code part in "else" is executed. 
# If break statement is executed inside for loop then the "else" part is skipped. 
# Note that "else" part is executed even if there is a continue statement.
# Prints out 0,1,2,3,4 and then it prints "count value reached 5"
count=0
while(count<5):
    print(count)
    count +=1
else:
    print("count value reached %d" %(count))

# Prints out 1,2,3,4
for i in range(1, 10):
    if(i%5==0):
        break
    print(i)
else:
    print("this is not printed because for loop is terminated because of break but not due to fail in condition")

#8 Functions
#What are Functions
# Functions are a convenient way to divide your code into useful blocks, 
# allowing us to order our code, make it more readable, reuse it and save some time. 
# Also functions are a key way to define interfaces so programmers can share their code.
 
#8.1 How do you write functions in Python?
# As we have seen on previous tutorials, Python makes use of blocks.
# A block is a area of code of written in the format of:

# block_head:
#     1st block line
#     2nd block line
#     ...

# Where a block line is more Python code (even another block), 
# and the block head is of the following format: block_keyword block_name(argument1,argument2, ...) 
# Block keywords you already know are "if", "for", and "while".
# Functions in python are defined using the block keyword "def", 
# followed with the function's name as the block's name. For example:
def my_function():
    print("Hello From My Function!")
    
#Functions may also receive arguments (variables passed from the caller to the function). 
#For example:
def my_function_with_args(username, greeting):
    print("Hello, %s , From My Function!, I wish you %s"%(username, greeting))

# Functions may return a value to the caller, using the keyword- 'return' . 
# For example:
def sum_two_numbers(a, b):
    return a + b

# 8.2 How do you call functions in Python?
# Simply write the function's name followed by (), placing any required arguments within the brackets. 
# For example, lets call the functions written above (in the previous example):
# Define our 3 functions
def my_function2():
    print("Hello From My Function!")

def my_function_with_args2(username, greeting):
    print("Hello, %s , From My Function!, I wish you %s"%(username, greeting))

def sum_two_numbers2(a, b):
    return a + b

# print(a simple greeting)
my_function2()

#prints - "Hello, John Doe, From My Function!, I wish you a great year!"
my_function_with_args2("John Doe", "a great year!")

# after this line x will hold the value 3!
x = sum_two_numbers2(1,2)
    
#9 Classes and Objects
# Objects are an encapsulation of variables and functions into a single entity. 
# Objects get their variables and functions from classes. Classes are essentially a template to create your objects.
# A very basic class would look something like this:
class MyClass:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")
        
# We'll explain why you have to include that "self" as a parameter a little bit later. 
# First, to assign the above class(template) to an object you would do the following:
class MyClass2:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass2()

# Now the variable "myobjectx" holds an object of the class "MyClass2" 
# that contains the variable and the function defined within the class called "MyClass".

# 9.1 Accessing Object Variables
#To access the variable inside of the newly created object "myobjectx" you would do the following:
class MyClass3:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass3()
myobjectx.variable
# So for instance the below would output the string "blah":
class MyClass4:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass4()
print(myobjectx.variable)

# You can create multiple different objects that are of the same class(have the same variables and functions defined). 
# However, each object contains independent copies of the variables defined in the class.
# For instance, if we were to define another object with the "MyClass" class and 
# then change the string in the variable above:
class MyClass5:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass5()
myobjecty = MyClass5()

myobjecty.variable = "yackity"

# Then print out both values
print(myobjectx.variable)
print(myobjecty.variable)

#9.2 Accessing Object Functions
# To access a function inside of an object you use notation similar to accessing a variable:
class MyClass6:
    variable = "blah"

    def function(self):
        print("This is a message inside the class.")

myobjectx = MyClass6()

myobjectx.function()

#10 Dictionary
# A dictionary is a data type similar to arrays, 
# but works with keys and values instead of indexes. 
# Each value stored in a dictionary can be accessed using a key,
#  which is any type of object (a string, a number, a list, etc.) instead of using its index to address it.

# For example, a database of phone numbers could be stored using a dictionary like this:
phonebook = {}
phonebook["John"] = 938477566
phonebook["Jack"] = 938377264
phonebook["Jill"] = 947662781
print(phonebook)

#Alternatively, a dictionary can be initialized with the same values in the following notation:
phonebook = {
    "John" : 938477566,
    "Jack" : 938377264,
    "Jill" : 947662781
}
print(phonebook)

#10.1 Iterating over dictionaries
# Dictionaries can be iterated over, just like a list. 
# However, a dictionary, unlike a list, does not keep the order of the values stored in it. 
# To iterate over key value pairs, use the following syntax:
phonebook = {"John" : 938477566,"Jack" : 938377264,"Jill" : 947662781}
for name, number in phonebook.items():
    print("Phone number of %s is %d" % (name, number))
    
#10.2 Removing a value
#To remove a specified index, use either one of the following notations:
phonebook = {
   "John" : 938477566,
   "Jack" : 938377264,
   "Jill" : 947662781
}
del phonebook["John"]
print(phonebook)

#or
phonebook = {
   "John" : 938477566,
   "Jack" : 938377264,
   "Jill" : 947662781
}
phonebook.pop("John")
print(phonebook)

#11 Modules and Packages
# In programming, a module is a piece of software that has a specific functionality. 
# For example, when building a ping pong game, one module would be responsible for the game logic, and
# another module would be responsible for drawing the game on the screen. 
# Each module is a different file, which can be edited separately.

#11.1 Writing modules
# Modules in Python are simply Python files with a .py extension. 
# The name of the module will be the name of the file. 
# A Python module can have a set of functions, classes or variables defined and implemented.
#  In the example above, we will have two files, we will have:

# mygame/
# mygame/game.py
# mygame/draw.py

# The Python script game.py will implement the game. 
# It will use the function draw_game from the file draw.py, or in other words, thedraw module, 
# that implements the logic for drawing the game on the screen.
# Modules are imported from other modules using the import command. 
# In this example, the game.py script may look something like this:

# game.py
# import the draw module
import draw

def play_game():
    ...

def main():
    result = play_game()
    draw.draw_game(result)

# this means that if this script is executed, then 
# main() will be executed
if __name__ == '__main__':
    main()
    
#The draw module may look something like this:
# draw.py

def draw_game():
    ...

def clear_screen(screen):
    ...

# In this example, the game module imports the draw module, 
# which enables it to use functions implemented in that module. 
# The main function would use the local function play_game to run the game,
# and then draw the result of the game using a function implemented in the draw module called draw_game.
# To use the function draw_game from the draw module, 
# we would need to specify in which module the function is implemented, using the dot operator.
# To reference the draw_game function from the game module, 
#  would need to import the draw module and only then call draw.draw_game().

# When the import draw directive will run, 
# the Python interpreter will look for a file in the directory which the script was executed from, 
# by the name of the module with a .py prefix, so in our case it will try to look for draw.py. 
# If it will find one, it will import it. If not, he will continue to look for built-in modules.

# You may have noticed that when importing a module, a .pyc file appears, 
# which is a compiled Python file. Python compiles files into Python bytecode 
# so that it won't have to parse the files each time modules are loaded. If a .pyc file exists, 
# it gets loaded instead of the .py file, but this process is transparent to the user.
    
#11.2 Importing module objects to the current namespace
# We may also import the function draw_game directly into the main script's namespace, by using the from command.
# game.py
# import the draw module
    
#THIS SHOULD Be Part of the Code
#from draw import draw_game
def main():
    result = play_game()
    draw_game(result)
    
# You may have noticed that in this example, draw_game does not precede with the name of the module it is imported from,
#  because we've specified the module name in the import command.

# The advantages of using this notation is that it is easier to use the functions inside the current module 
# because you don't need to specify which module the function comes from. 
# However, any namespace cannot have two objects with the exact same name,
# so the import command may replace an existing object in the namespace.
    
#11.3 Importing all objects from a module
#We may also use the import * command to import all objects from a specific module, like this:
# game.py
# import the draw module
#THIS SHOULD Be Part of the Code
#from draw import *

def main():
    result = play_game()
    draw_game(result)
    
# This might be a bit risky as changes in the module might affect the module which imports it,
# but it is shorter and also does not require you to specify which objects you wish to import from the module.

#11.4 Exploring built-in modules

# Check out the full list of built-in modules in the Python standard library here.
# Two very important functions come in handy when exploring modules in Python - the dir and help functions.
# If we want to import the module urllib, which enables us to create read data from URLs, we simply import the module:
# import the library
import urllib
# use it
urllib.urlopen(...)

#12 Numpy Arrays
#12.1 Getting started

# Numpy arrays are great alternatives to Python Lists. 
# Some of the key advantages of Numpy arrays are that they are fast, easy to work with, 
# and give users the opportunity to perform calculations across entire arrays.
# In the following example, you will first create two Python lists. 
# Then, you will import the numpy package and create numpy arrays out of the newly created lists.

# Create 2 new lists height and weight
height = [1.87,  1.87, 1.82, 1.91, 1.90, 1.85]
weight = [81.65, 97.52, 95.25, 92.98, 86.18, 88.45]

# Import the numpy package as np
import numpy as np

# Create 2 numpy arrays from height and weight
np_height = np.array(height)
np_weight = np.array(weight)

#12.2 Element-wise calculations
# Now we can perform element-wise calculations on height and weight.
# For example, you could take all 6 of the height and weight observations above, 
# and calculate the BMI for each observation with a single equation. 
# These operations are very fast and computationally efficient. 
# They are particularly helpful when you have 1000s of observations in your data.
# Calculate bmi
bmi = np_weight / np_height ** 2
# Print the result
print(bmi)

#12.3 Subsetting

# Another great feature of Numpy arrays is the ability to subset. 
# For instance, if you wanted to know which observations in our BMI array are above 23, 
# we could quickly subset it to find out.
# For a boolean response
bmi > 23

# Print only those observations above 23
bmi[bmi > 23]

#13 Pandas Basics
#13.1 Pandas DataFrames

# Pandas is a high-level data manipulation tool developed by Wes McKinney. 
# It is built on the Numpy package and its key data structure is called the DataFrame. 
# DataFrames allow you to store and manipulate tabular data in rows of observations and columns of variables.
# There are several ways to create a DataFrame. One way way is to use a dictionary. For example:
dict = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }

import pandas as pd
brics = pd.DataFrame(dict)
print(brics)

# As you can see with the new brics DataFrame, 
# Pandas has assigned a key for each country as the numerical values 0 through 4. 
# If you would like to have different index values, say, the two letter country code, 
# you can do that easily as well.

# Set the index for brics
brics.index = ["BR", "RU", "IN", "CH", "SA"]

# Print out brics with new index values
print(brics)

# Another way to create a DataFrame is by importing a csv file using Pandas. 
# Now, the csv cars.csv is stored and can be imported using pd.read_csv:
# Import pandas as pd
import pandas as pd

# Import the cars.csv data: cars
cars = pd.read_csv('cars.csv')

# Print out cars
print(cars)

#13.2 Indexing DataFrames

# There are several ways to index a Pandas DataFrame. 
# One of the easiest ways to do this is by using square bracket notation.

# In the example below, you can use square brackets to select one column of the cars DataFrame. 
# You can either use a single bracket or a double bracket. The single bracket with output a Pandas Series, 
# while a double bracket will output a Pandas DataFrame.

# Import pandas and cars.csv
import pandas as pd
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out country column as Pandas Series
print(cars['cars_per_cap'])

# Print out country column as Pandas DataFrame
print(cars[['cars_per_cap']])

# Print out DataFrame with country and drives_right columns
print(cars[['cars_per_cap', 'country']])

#Square brackets can also be used to access observations (rows) from a DataFrame. For example:
# Import cars data
cars = pd.read_csv('cars.csv', index_col = 0)

# Print out first 4 observations
print(cars[0:4])

# Print out fifth and sixth observation
print(cars[4:6])

# You can also use loc and iloc to perform just about any data selection operation. loc is label-based, 
# which means that you have to specify rows and columns based on their row 
# and column labels. iloc is integer index based, so you have to specify rows 
# and columns by their integer index like you did in the previous exercise.

cars = pd.read_csv('cars.csv', index_col = 0)

# Print out observation for Japan
print(cars.iloc[2])

# Print out observations for Australia and Egypt
print(cars.loc[['AUS', 'EG']])