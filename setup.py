from setuptools import setup, find_packages

setup(
    name="stampli",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic",
        "langchain",
        "langchain-openai",
        "requests",
        "pytest",
        "asyncio",
        "ipython",
        "ipywidgets",
    ],
    python_requires=">=3.9",
)
