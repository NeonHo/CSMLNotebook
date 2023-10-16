Assuming a is a string. The Slice notation in python has the syntax -
```Python
list[<start>:<stop>:<step>]
```

So, when you do `a[::-1]`, it starts from the `end` towards the `first` taking each element. So it reverses `a`. This is applicable for lists/tuples as well.

Example -
```Python
>>> a = '1234'
>>> a[::-1]
'4321'
```
Then you convert it to int and then back to string (Though not sure why you do that) , that just gives you back the string.
