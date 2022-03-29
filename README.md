[![doc](https://img.shields.io/badge/-Documentation-blue)](https://advestis.github.io/RIPE)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

#### Status
![push](https://github.com/Advestis/RIPE/actions/workflows/push.yml/badge.svg)

![maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![issues](https://img.shields.io/github/issues/Advestis/RIPE.svg)
![pr](https://img.shields.io/github/issues-pr/Advestis/RIPE.svg)


#### Compatibilities
![ubuntu](https://img.shields.io/badge/Ubuntu-supported--tested-success)
![unix](https://img.shields.io/badge/Other%20Unix-supported--untested-yellow)

![python](https://img.shields.io/pypi/pyversions/RIPE)


##### Contact
[![linkedin](https://img.shields.io/badge/LinkedIn-Advestis-blue)](https://www.linkedin.com/company/advestis/)
[![website](https://img.shields.io/badge/website-Advestis.com-blue)](https://www.advestis.com/)
[![mail](https://img.shields.io/badge/mail-maintainers-blue)](mailto:pythondev@advestis.com)

# RIPE
Implementation of a rule based prediction algorithm called RIPE (Rule Induction Partitioning Estimate). RIPE is a deterministic and interpretable algorithm, for regression problem. It has been presented at the International Conference on Machine Learning and Data Mining in Pattern Recognition 2018 (MLDM 18). The paper is available in arXiv https://arxiv.org/abs/1807.04602.

## Getting Started
These instructions will get you a copy of the project up and running on your
local machine for development and testing purposes. See deployment for notes
on how to deploy the project on a live system.

### Prerequisites
RIPE is developed in Python version 2.7. It requires some usual packages
- NumPy (post 1.13.0)
- Scikit-Learn (post 0.19.0)
- Pandas (post 0.16.0)
- SciPy (post 1.0.0) 
- Matplotlib (post 2.0.2) 
- Seaborn (post 0.8.1)

See **requirements.txt**.
```
sudo pip install package_name
```
To install a specific version
```
sudo pip install package_name==version
```

### Installing

The latest version can be installed from the master branch using pip:
```
pip install git+git://github.com/VMargot/RIPE.git
```
Another option is to clone the repository and install using ```python setup.py
install``` or ```python setup.py develop```.

## Usage
RIPE has been developed to be used as a regressor from the package scikit-learn.

### Training
```
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data, iris.target

ripe = RIPE.Learning()
ripe.fit(X, y)
```

### Predict
```
ripe.predict(X)
```

### Score
```
ripe.score(X,y)
```

### Inspect rules:
To have the Pandas DataFrame of the selected rules
```
ripe.selected_rs.to_df()
```
Or, one can use
```
ripe.make_selected_df()
```
To draw the distance between selected rules
```
ripe.plot_dist()
```
To draw the count of occurrence of variables in the selected rules
```
ripe.plot_counter_variables()
```

## Notes
This implementation is in progress. If you find a bug, or something witch could
be improve don't hesitate to contact me.

## Authors
* **Vincent Margot**

See also the list of [contributors](https://github.com/VMargot/RIPE/contributors)
who participated in this project.

## License

This project is licensed under the GNU v3.0 - see the [LICENSE.md](LICENSE.md)
file for details
