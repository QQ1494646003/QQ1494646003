## Transformer Problem Classification system based on NLP:Keyword Extraction
As we all known, whether it is a course quiz problem or some algorithm problem like leetcode, the explanation of the problem itself is usually not a sentence, it  tends to be in the form of a small paragraph. 
Using Natural Language Processing method to classify the problems in large quantities into categories based on keywords is now a hot direction in the industry. Compared to traditional methods, NLP based method can save both a lot of labor costs and time. 

## Before you run
Prepare CUDA+CUDNN Computing Environment, Anaconda with Python==3.6.

Prepare python environment:
`pip install -r requirement.txt`

All right to go.

### Peak-View of Dataset
`python DataSpider.py` to Generate a Peak-View of Dataset, set detail mode to False in "main" of that script for simpler information.

### Run

`python dashboard.py`

Use Enterprise token as 123456(Enterprise License). (saved in enterprise_key.token) Or type "Enter" to skip this(Community License).

1. One-shot Inference stands for inference one question at one time, means extract the information keywords for one question.
2. Batch Inference, inference many questions at a time, used to create the problem-feature dataset. You NEED to clean historical inference data before you run, which is Output/question_DB.csv. Delete it before you run.(Enterprise License)
3. Get problems by keyword allow you to search for questions using only one keyword.
4. Get problems by keywords allow you to search for questions using multiple keywords.(Enterprise License) The result will be saved in "Output/".



That's all folks.
