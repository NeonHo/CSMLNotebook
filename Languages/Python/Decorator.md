Decorators are functions that take another function as an argument and return a new function with added functionality.

With a decorator, we can attach pre- or post-processing operations to an existing function.

Before the definition of the function, we want to decorate, we place the name of the decorator function with a leading `@` symbol.

```Python
@decorator_name
def decorated_func(args):
	return args
```

Decorator can be a function or class.

The code above is equal to the code as follows:

```Python
def decorated_func(args):
	return args

decorated_func = decorator_name(decorated_func)
```

# `property()`

```Python
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property  # getter
    def radius(self):
        """The radius property."""
        print("Get radius")
        return self._radius

    @radius.setter
    def radius(self, value):
        print("Set radius")
        self._radius = value

    @radius.deleter
    def radius(self):
        print("Delete radius")
        del self._radius

```

We can look up the methods using `dir` [[dir()]].
```Python
>>> dir(Circle.radius)
[..., 'deleter', ..., 'getter', 'setter']
```

We can use these methods:

```Python
>>> from circle import Circle

>>> circle = Circle(42.0)

>>> circle.radius
Get radius
42.0

>>> circle.radius = 100.0
Set radius
>>> circle.radius
Get radius
100.0

>>> del circle.radius
Delete radius
>>> circle.radius
Get radius
Traceback (most recent call last):
    ...
AttributeError: 'Circle' object has no attribute '_radius'
```
## 何时创建
具体而言，当你使用类来创建一个新的对象时，Python会首先调用类的 __new__() 方法来创建对象的实例，然后调用 __init__() 方法来初始化该实例的属性。
属性是在类实例化时创建的，并在 __init__() 方法中进行了初始化。
# Note
