# KotlinCodeCompletion
to run program beside imported libraries:
-download pyarrow
-use python 3.8.2 or any other compatible with torch
before running it for the first time enable developer mode
(for model downloading purpose)

-clone external package for evaluationn
git clone https://github.com/amazon-science/mxeval.git
pip install -e mxeval

-accelerate > 0.26

changes in mxeval:
file: execution.py
line 505 add shell = True to subprocess parameters
line 501 add encoding='utf-8'

