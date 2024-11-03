# KotlinCodeCompletion
to run program beside imported libraries:<br>
-download pyarrow<br>
-use python 3.8.2 or any other compatible with torch<br>
before running it for the first time enable developer mode<br>
(for model downloading purpose)<br>

-clone external package for evaluationn<br>
git clone https://github.com/amazon-science/mxeval.git<br>
pip install -e mxeval<br>

-accelerate > 0.26<br>

changes in mxeval:<br>
file: execution.py<br>
line 505 add shell = True to subprocess parameters<br>
line 501 add encoding='utf-8'

