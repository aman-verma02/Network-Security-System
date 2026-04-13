'''
`setup.py` plays an important role in the Python project lifecycle by enabling your project to be packaged, distributed, and installed in a standardized way. It acts as a configuration file that tells Python how your project is structured, what dependencies it requires, and how it should be installed. This is especially useful in larger applications like machine learning or cybersecurity projects, where code needs to be reusable, modular, and easy to deploy across different environments such as local machines, Docker containers, or cloud platforms. By using `setup.py`, developers can treat their own code like a proper Python package, improving maintainability and scalability.

### 📌 Key Uses of `setup.py`

* 📦 **Packaging the project**: Converts your code into an installable Python package.
* 📚 **Dependency management**: Automatically installs required libraries like NumPy, Pandas, and Scikit-learn.
* 🔁 **Code reusability**: Allows importing modules cleanly across different parts of the project.
* 🚀 **Simplifies deployment**: Makes it easier to set up the project in Docker or cloud environments.
* 🧪 **Version control**: Helps manage versions of your project (e.g., 1.0.0, 2.0.0).
* 🔗 **CLI support**: Enables creation of custom command-line tools.

In modern development, while newer tools like `pyproject.toml` are becoming popular, `setup.py` is still widely used and remains highly relevant, especially in machine learning and production-grade Python applications.


'''


from setuptools import setup, find_packages
from typing import List

def get_requirements()->List[str]:
    '''
    This function will return the list of requirements
    ''' 

    requirement_list:List[str] = []
    try: 
        with open('requirements.txt', 'r') as file:
            #Read lines from the file
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement!= '-e .':
                    requirement_list.append(requirement)
                
    except FileNotFoundError:
        print("requirement.txt file not found.")

    return requirement_list



setup(
    name='NetworkSecurity',
    version='0.0.1',
    author="Mac Trek",
    author_email="aksvbpl@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)

