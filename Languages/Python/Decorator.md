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

decorated_func = decorator_name()
```