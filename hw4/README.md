# File Structure

```
.
├── README.md
├── __init__.py
├── build
│   ├── bdist.macosx-10.9-x86_64
│   └── lib
│       └── __init__.py
├── data
│   ├── iris
│   │   ├── README.md
│   │   ├── format.txt
│   │   ├── test.txt
│   │   └── train.txt
│   └── mnist
│       ├── README.md
│       ├── format.txt
│       ├── test.txt
│       └── train.txt
├── figs
│   └── mnist_elbow_curve.png
├── outputs
│   ├── 1337_iris_labels.txt
│   ├── 1337_kmeans_advanced_iris_labels.txt
│   ├── 1337_mnist_advanced_labels.txt
│   ├── 1337_mnist_basic_labels.txt
│   ├── 1_kmeans_advanced_iris_labels.txt
│   ├── 42_PCA_75_mnist_advanced_labels.txt
│   ├── 42_PCA_85_mnist_advanced_labels.txt
│   ├── 42_PCA_90_mnist_advanced_labels.txt
│   ├── 42_PCA_95_mnist_advanced_labels.txt
│   ├── 42_PCA_99_mnist_advanced_labels.txt
│   ├── 42_UMAP_chebyshev_mnist_advanced_labels.txt
│   ├── 42_UMAP_mnist_advanced_labels.txt
│   ├── 42_forgy_kmeans_advanced_iris_labels.txt
│   ├── 42_iris_labels.txt
│   ├── 42_kmeans++_PCA_0.9_kmeans_advanced_iris_labels.txt
│   ├── 42_kmeans++_UMAP_density_kmeans_advanced_iris_labels.txt
│   ├── 42_kmeans++_UMAP_kmeans_advanced_iris_labels.txt
│   ├── 42_kmeans_advanced_iris_labels.txt
│   ├── 42_mnist_advanced_labels.txt
│   ├── 42_mnist_basic_labels.txt
│   └── iris_labels.txt
├── setup.py
└── src
    ├── __init__.py
    ├── __pycache__
    │   ├── kmeans.cpython-310.pyc
    │   ├── kmeans_advanced.cpython-310.pyc
    │   ├── load_iris.cpython-310.pyc
    │   └── load_mnist.cpython-310.pyc
    ├── elbow_curve.py
    ├── hw4.egg-info
    │   ├── PKG-INFO
    │   ├── SOURCES.txt
    │   ├── dependency_links.txt
    │   └── top_level.txt
    ├── iris.py
    ├── kmeans.py
    ├── load_iris.py
    ├── load_mnist.py
    └── mnist.py

11 directories, 49 files
```

# How to run

```
cd src
python iris.py
python mnist.py
python elbow_curve.py
```

- ```load_iris.py``` implements IRIS loading functionality.
- ```load_mnist.py``` implements MNIST loading functionality.
- ```kmeans.py``` implements KMeans Clustering algorithm.
- ```iris.py``` clusters given IRIS data using implemented KMeans.
- ```mnist.py``` clusters given MNIST data using implemented KMeans.
- ```elbow_curve.py``` plots elbow curve for MNIST data.

- ```iris.py``` and ```mnist.py``` will generate output files in ```outputs```
folder.
- ```elbow_curve.py``` will generate ```mnist_elbow_curve.png``` in ```figs```

The current configs in the file correspond to my most recent submission on
miner, which is also my highest score.
