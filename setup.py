from setuptools import setup, find_packages

setup(
    name="efiras",
    version="0.1.0",
    packages=find_packages(where=".", include=["core*", "database*", "utils*", "auth*"]),  # Find your actual packages
    package_dir={"": "."},
    
    # Add minimum requirements to avoid issues
    install_requires=[
        "fastapi",
        "uvicorn",
        "sqlalchemy", 
        "psycopg2-binary",
        "python-dotenv",
    ],
    
    # Python version requirement
    python_requires=">=3.8",
)