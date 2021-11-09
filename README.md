# FakeNewsDetection

This is a project taken up in spare time to test a simple Machine Learning algorithm using the functionality of sklearn along with the pandas funciton in Python to be used as a dataframe.

The dataset used in this is provided from data-flair and hosted on a Google Drive.

A few useful things to mention is the usage of the TfidfVectorizer.  TF-IDF stands for "Term Frequency - Inverse Document Frequency".  This is designed to place weight on certain words in a document based upon frequency of occurrence.  More can be found out about it here:
https://monkeylearn.com/blog/what-is-tf-idf/

Last but not least is the confusion matrix.  This takes the values of the test data in this case and compares it to what was predicted based upon the code itself and displays it as a grid showing the Positive/Negatives.  The True Positive would be in the upper left while the True Negative would be in the lower right.
