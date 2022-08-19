# Association rule mining
This is a course final project, presenting a solution with a purpose of extracting rules from dataset of transactions. Solution has been tested on specific domain related to persons prefereces in manga and comics.

More about this method of knowledge extraction can be found here: https://en.wikipedia.org/wiki/Association_rule_learning

The provided solution uses **Apriori algorithm** for mining rules from data and presents them in various forms (Popular sets, Rules, If->Then).


# Local installation and running

To run this project at your machine you need install required packages from **requirements.txt**. This can be done with following command.:

```bash
pip install -r requirements.txt
```
Prior to that I recommend to create your own venv. After that you can launch server locally with this command:

```bash
streamlit run main.py
```
The webpage will open and interactable UI will appear.