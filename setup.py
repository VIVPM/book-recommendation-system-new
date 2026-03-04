from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "book-recommendation-system-new"
AUTHOR_USER_NAME = "VIVEK P M"
SRC_REPO = "book_recommender"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="VIVEK P M",
    description="A small local packages for ML based books recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VIVPM/book-recommendation-system-new",
    author_email="vivpm99@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.9",
    install_requires=LIST_OF_REQUIREMENTS
)