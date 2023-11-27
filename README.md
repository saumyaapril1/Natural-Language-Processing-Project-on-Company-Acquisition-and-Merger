# Natural-Language-Processing-Project-on-Company-Acquisition-and-Merger


ABSTRACT

Since the time unknown to today’s era, as the man is developing and moving towards technology, the data is increasing at an exponential rate. And this data has huge potential in its own if processed and information is derived successfully, which is the biggest challenge. To combat and overcome this challenge, research is seen in Data Science with the help of Machine learning. And Natural Language Processing is a part of this. Natural Language Processing (NLP) is an effort to derive useful information from the data in the form of text. If NLP is successfully implemented, it is potent to save lots of time and increase the rate of actions that need to be taken after data analysis. Be it the field of finance and economy, medical or any other, NLP can prove to be a boon. And through our project, we tried to do our bit by taking the problem of deriving information about company acquisition data available in the form of the text.The dataset used is the headlines from the newspaper and other news source. The  statement contains news about acquisition of a company, by a company and it predicts about 
[ Acquired_Final ] & [ Acquired_Target ] using Natural Language Processing, where
[ Acquired_Final ] denotes the body which acquires and [ Acquired_Target ] refers to the acquired body.


INTRODUCTION

A country’s economy plays a very important role in deciding its position in the world. 
Today’s economic market is highly competitive and financially constrained. The corporate sector must identify and work on growth strategies which can improve and strengthen their position in this competitive world, where it takes seconds to get left out in the race. The incentives and measures towards strengthening their positions include expansion of their enterprise, entering joint ventures, merging or acquiring the existing ventures of their competitors in the market.

Acquisitions and merging can dramatically change the status of an organisation. An “on the verge of failing venture” can boom if right decisions of acquisition or merge are taken.

And this process of acquisition and merging have been taking place all over the financial market between several ventures. This determines the state of the economy of the country to a pretty extent. Analysing the activity of acquiring-merging and drawing pertinent results can be very fruitful for calculating the economy parameters and other significant figures. 
What if we try to speed up the process by automating the task of company acquisition analysis ? It will not only save cost but also save time, by giving room for other research and development in the same.

Data Source:
Dataset was taken from the data provided at the hackathon organised by Global Institute of technology. 
The dataset is balanced dataset.
There are three datasets being taken:
Corpus.xlsx - 900 rows*6 columns
This has the following features: system_id, title, org_link, source, date, raw_article.

MnA_Training.xlsx - 500 rows*3 columns
MnA_Test.xlsx - Test data 250 rows*3 columns
This has the following features: system_id, Acq_Final and Tar_Final



RELATED WORK:


Approach used is Rule-Based approach which always depend on dataset and hence no research paper is published in the field after 2015. This approach is not widely used as it produces specific result corresponding only to the dataset and any change in context or pre-text to the dataset will void the functionality of the running approach. The new model uses Word2Vec form of computing but to our specific dataset the Rule-Based-Approach is providing satisfactory result and thus need for implementing Word2Vec is not required. Some of the research paper which aim on getting the same results are mentioned below.

Using NLP on news headlines to predict index trends
In this paper, we have realised an attempt to show the prediction trend using just the headlines of the news rather than the entire article. The research focuses on the prediction of DJIA1 trends using NLP. Many possible algorithms and various embedding techniques have been tried and tested to come to an optimal technique. It primarily aims at using deep learning models to extract information from the corpus.


Opinion mining on newspaper headlines using NLP & SVM 
Opinion Mining is just another phrase used for the Sentiment Analysis, which is a technique that uses Natural Language processing (NLP) in classification of the text into outcome. For text data processing, there are various NLP tools used during this research, which have been done in opinion mining for online social networking sites like Twitter, Facebook, Linkedin, online blogs, etc. This research paper has proposed a new sentiment analysis technique using Support vector machine and NLP tools on headlines in Newspaper. On comparing the three models using confusion matrix, it has been concluded that Tf-idf and linear SVM provides relatively better accuracy than other models, but this is limited to smaller dataset only. In case of larger dataset, SGD and linear SVM model performs the best amongst the other models that are implemented during the research.


METHODOLOGY:

Initially, our approach was limited to the following steps followed by BOW (Bag Of Words) :
Tokenizing - the procedure of tokenizing or parting a string, content into a rundown of tokens. One can consider token parts like a word is a token in a sentence, and a sentence is a token in a section.
Stemming - Stemming is the way toward decreasing a word to its promise stem that fastens to additions and prefixes or to the underlying foundations of words known as a lemma.

Tagging -   POS labelling is the way toward increasing a word in a corpus to compare some portion of a discourse tag, in light of its unique circumstance and definition.
Bag of words - The pack of-words model is an improving portrayal utilized in characteristic language handling and data recovery. In this model, content is spoken to as the pack of its words, dismissing sentence structure and word request yet keeping assortment.

But we were unable to get a decent accuracy, due to which we had to change the course of our action, and switch to better approach that is, the Rule Based Approach. 
Rule based approach:
Library used https://stanfordnlp.github.io/CoreNLP/ which is better than NLTK library as it provides customization to pre-process text using Parser, Tagger, Tokenizer and PorterStemmer.

Rule-based approaches are one of the oldest approaches to NLP, which are safe because they are tired and have been proven to work well. 
In this, rules are applied to text and the meaning is also fed to the model.
Some examples of rule-based approaches to NLP are the context free grammars and regular expressions.
Rule-based approaches are low precision, high recall.
In the python code, some of the significant modules that have been imported and used are:




from isc_tokenizer import Tokenizer
tk = Tokenizer(lang='en')


from isc_tagger import Tagger
tagger = Tagger(lang='eng')


from isc_parser import Parser
parser = Parser(lang='eng')


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


Tokenizer:
The tokenize module provides a lexical scanner for Python source code, which is implemented in Python. The scanner in this module returns text as tokens.  


Tagger:
It reads text in a respective language and assign parts of speech to each token (noun, adjective, verb, adverb, etc.)


Parser:
It determines the syntactic structure of text by taking every word and the underlying grammar into account.
Stemmer:
It reduces the word to its root/base word. For instance, after stemming both the words: softly and softness will be reduced to soft. 


The approach:
Based on the code, the method implemented to get the solution is as follows:
The following is the standard approach followed by the rule based approach:
Matches the unique id from the corpus and the training set.
Then the tokenizer splits the data into isolated figments, known as tokens.
The tagger assigns part of speech to the tokenized text (the tokens).
Parser figures out the dependence of words on each other, determining the syntactic structure. 
Proper Noun:
Buy: Token after “buy” is target
Sell: Token after “sell” is target
Sells: Token after “sells” is target
Sells to: Token after “sells to” is acquiring
At last, the unique id is matched from the corpus and test set and predictAcq_Final & Tar_Final according to the result of the algorithm.




Inputs:
The inputs are the news headlines along with the system ids that were provided with the dataset. Any of the random generated system_id can be taken as input to test with the model.



Output(s):
We used a loop to make this model run for each and every news headline in the dataset and save it in an excel file corresponding to the same system_id.


FUTURE SCOPE:
We have just taken the headlines from the news, rather than the entire article as the data in the article is not cleaned. So, we can consider working on cleaning the text in the article and analysing the article as well.
Rule based approach has high recall and low precision, which implies that they can have high performance in specific use cases, but can degrade the performance when generalised.
We will be taking neural networks into account as with the help of neural networks, we can increase the accuracy and expect accurate prediction of more generalised data, which will make our approach more flexible rather than dataset centric.


REFERENCES:

[1].Yano, Tae, Noah A. Smith, and John D. Wilkerson, 2012, Textual predictors of bill     survival in Congressional committees, in Proceedings of the Conference of the North American Chapter of the Association for Computational Linguistics pp. 793–802 Montreal, Quebec. 

[2].Yano, Tae, Dani Yogatama, and Noah A. Smith, 2013, A penny for your tweets: Campaign contributions and Capitol Hill microblogs, in Proceedings of the International AAAI Conference on Weblogs and Social Media Boston, MA.

[3].Buehlmaier, Matthias M. M., and Toni M. Whited, 2014, Looking for risk in words: A narrative approach to measuring the pricing implications of financial constraints, Working PaperHoberg, Gerard, and Vojislav Maksimovic, 2015, Redefining financial constraints: A text based analysis, Review of Financial Studies 28, 1312–1352

[4].Hoberg, Gerard, and Vojislav Maksimovic, 2015, Redefining financial constraints: A text based analysis, Review of Financial Studies 28, 1312–1352



