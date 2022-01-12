# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['magnus', 'magnus.executor_extensions']

package_data = \
{'': ['*']}

install_requires = \
['dataclasses-json', 'ruamel.yaml', 'ruamel.yaml.clib', 'yachalk']

extras_require = \
{'docker': ['docker']}

entry_points = \
{'console_scripts': ['magnus = magnus.cli:main']}

setup_kwargs = {
    'name': 'magnus',
    'version': '0.1.0',
    'description': 'A Compute agnostic pipelining software',
    'long_description': None,
    'author': 'Vijay Vammi',
    'author_email': 'vijay.vammi@astrazeneca.com',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'extras_require': extras_require,
    'entry_points': entry_points,
    'python_requires': '>=3.7,<4.0',
}


setup(**setup_kwargs)
