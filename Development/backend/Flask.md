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
``

