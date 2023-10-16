Require a bit of boilerplate up front:
1. avoid common pitfalls (likely mistakes in a situation)
2. create a project that is easy to expand on.

# Application Setup
A Flask application: an instance of the Flask class.
1. configurations
2. URLs
The most straightforward way to create a Flask application is to create a global Flask instance directly at the top of your code.
```Python
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'
```
## Application Factory
### Create the flaskr directory and add the `__init__.py` file.
The __init__.py serves double duty: 
1. it will contain the application factory, 
2. and it tells Python that the flaskr directory should be treated as a package.


# URL定义
URL 是由你自己定义的，它是你的 Flask 后端应用程序中的路由。在 Flask 中，你可以通过使用 `@app.route()` 装饰器来定义路由，这决定了在哪个URL上可以访问你的应用程序的特定功能。

在之前的示例中，我们定义了一个 `/quant_config_modify` 的路由，它接受 POST 请求。这意味着你的应用程序将在 `http://127.0.0.1/quant_config_modify` 上监听 POST 请求，并通过 `quant_config_modify` 函数来处理这些请求。

你可以根据你的项目需求自定义路由，但需要确保路由的唯一性和明确性，以避免与其他路由冲突。通常，路由可以反映应用程序的功能或资源，以便通过不同的URL访问不同的功能。

在 Flask 中，路由可以包含变量，例如：

```python
@app.route('/user/<username>')
def show_user_profile(username):
    # 在这里根据用户名显示用户的个人资料
    return 'User Profile: {}'.format(username)
```

在上述示例中，`/user/<username>` 路由包含一个变量 `<username>`，它允许你访问不同用户的个人资料。这个变量的值将作为参数传递给路由处理函数。

总之，URL 是由你在 Flask 应用程序中定义的路由来决定的，你可以根据你的需求自定义路由以映射到不同的功能或资源。

# 传给Vue的可以直接用于vue的list

flask后端应该如何返回才能得到下述这种Vue的数组?
```Python
		options: [{
          value: '选项1',
          label: '黄金糕'
        }, {
          value: '选项2',
          label: '双皮奶'
        }, {
          value: '选项3',
          label: '蚵仔煎'
        }, {
          value: '选项4',
          label: '龙须面'
        }, {
          value: '选项5',
          label: '北京烤鸭'
        }],
```
```