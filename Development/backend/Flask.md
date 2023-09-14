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
### `__init__.py`
