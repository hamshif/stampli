from setuptools import setup, find_packages

setup(
    name="stampli",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Core scvi-tools dependencies
        "scvi-tools>=1.0.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",

        # Deep learning frameworks
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "transformers>=4.21.0",
        "sacremoses>=0.0.53",

        # Data manipulation and analysis
        "pandas>=2.2.2",
        "openpyxl>=3.1.5",
        "numpy>=2.1.1",
        "scipy>=1.9.0",
        "scikit-learn>=1.1.0",
        "pyspark>=3.5.0",
        "pyarrow", 

        # Visualization
        "matplotlib>=3.9.2",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "pillow>=10.4.0",
        "exifread>=3.1.0",
        "ffmpeg>=1.4.0",

        # Jupyter notebook dependencies
        "ipython>=8.28.0",
        "ipykernel>=6.29.5",
        "jupyter>=1.0.0",
        "jupyterlab>=4.2.5",
        "notebook>=7.2.2",
        "ipywidgets>=8.1.5",
        "jupyter-contrib-nbextensions>=0.7.0",

        # Progress bars and utilities
        "tqdm>=4.66.5",
        "rich>=13.9.1",

        # Web backend and LLM orchestration
        "requests>=2.32.3",
        "httpx>=0.27.2",
        "fastapi>=0.115.2",
        "uvicorn[standard]>=0.30.6",
        "pydantic>=2.9.2",
        
        # Stampli Specific
        "langchain",
        "langchain-openai",
    ],
    python_requires=">=3.9",
)
