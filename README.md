# Sentiment Analysis


In this lesson, we're going to learn how to use [VADER](https://github.com/cjhutto/vaderSentiment), an English-language sentiment analysis tool designed for use with social media texts. This tool helps to calculate whether texts express positive or negative sentiment.

In this lesson, we will learn how to use VADER to:
- Calculate sentiment for individual sentences, tweets, and a fairy tale
- Make plots of how sentiment fluctuates over time and throughout a text
---

## Acknowledgement

The following tutorial is almost entirely copied from Melanie Walsh's [Sentiment Analysis lesson](https://melaniewalsh.github.io/Intro-Cultural-Analytics/05-Text-Analysis/04-Sentiment-Analysis.html). This tutorial differs from Walsh's lesson primarily in the datasets used. Additional steps were added for working with JSON data and web scraping a text file.

---

## Datasets

We are going to analyze tweets related to the Biden Administration's student loan forginess program. You can download the dataset [here](https://drive.google.com/drive/u/0/folders/15MC7UJd5Sz0hSKqY2XsFUbFbAM-t065a).

We are also going to analyze President Biden's 2023 State of the Union Address, which we will web scrape from the White House website. 



## What is Sentiment? What Exactly Are We Measuring?

What is sentiment analysis, exactly? What are we actually measuring with VADER?

These are tough and important questions. According to VADER's creators, C.J. Hutto and Eric Gilbert, "Sentiment analysis, or opinion mining, is an active area of study in the field of natural language processing that analyzes people's **opinions, sentiments, evaluations, attitudes, and emotions** via the computational treatment of subjectivity in text" (["VADER"](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109/8122)).

I would like to emphasize that "opinions, sentiments, evaluations, attitudes, and emotions" covers a *lot* of ground. These are complex categories of human experience that can be expressed in many different ways, especially in different contexts. In fact, that's why Hutto and Gilbert designed VADER specifically for *social media* texts, because sentiment gets expressed differently in social media posts than in, say, fictional stories or newspaper articles.

We want to remain critical and self-reflexive about what exactly we are measuring with VADER — especially when we apply VADER to non-social media texts, as we will with "Little Red-Cap" below. Too often I have encountered well-meaning students who want to understand complex social or textual phenomena (e.g., mental health, a reader's emotional experience of a story, opinions about climate change) by reducing it only to sentiment analysis scores, without doing more and without thinking more deeply about whether those scores actually capture what they're interested in.

So, yes, tools like VADER can be useful, as we will see below, but they are only useful when thoughtfully and deliberately applied.

## How VADER Was Built and How It Works

VADER, which stands for **V**alence **A**ware **D**ictionary and s**E**ntiment **R**easoner, calculates the sentiment of texts by referring to a lexicon of words that have been assigned sentiment scores as well as by using a handful of simple rules.

You can read more about how VADER was designed in [C.J. Hutto and Eric Gilbert's published paper](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109/8122), but here's a summary version: VADER's lexicon was created by enlisting 10 different people to rate thousands of words positively or negatively on a scale of -4 to 4 (you can scroll through the [lexicon on GitHub](https://github.com/cjhutto/vaderSentiment/blob/master/vaderSentiment/vader_lexicon.txt) and check it out for yourself — each line consists of a word, the mean rating, standard deviation, and 10 individual human ratings).

[![](../images/VADER-fig.png)](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109/8122)

*This figure shows the interface presented to the 10 raters for rating the sentiment of words. It is taken from ["VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text."](https://www.aaai.org/ocs/index.php/ICWSM/ICWSM14/paper/view/8109/8122)* 

For example, you can see that the acronym [ROFL](https://github.com/cjhutto/vaderSentiment/blob/d8da3e21374a57201b557a4c91ac4dc411a08fed/vaderSentiment/vader_lexicon.txt#L368) (Rolling on the Floor Laughing) has a mean rating of 2.7, the word [*crappy*](https://github.com/cjhutto/vaderSentiment/blob/d8da3e21374a57201b557a4c91ac4dc411a08fed/vaderSentiment/vader_lexicon.txt#L1622) has a mean rating of -2.5, and the word [*bliss*](https://github.com/cjhutto/vaderSentiment/blob/d8da3e21374a57201b557a4c91ac4dc411a08fed/vaderSentiment/vader_lexicon.txt#L1127) has a mean rating of 2.7. If you look at the 10 individual ratings for each entry, however, you can see interesting discrepancies. One person rated *bliss* as a 4, while another person rated *bliss* as a 1. Just how "positive" is the word *bliss*? What do *you* think?

What about [*cutesie*](https://github.com/cjhutto/vaderSentiment/blob/d8da3e21374a57201b557a4c91ac4dc411a08fed/vaderSentiment/vader_lexicon.txt#L1721)? It has a mean rating of 1, but two people rated it as a -1, and five people rated as a 2. Is *cutesie* an admiring adjective — "[She was so cutesie I just had to talk to her](https://www.urbandictionary.com/define.php?term=Cutesie)" — or a diminutive slight — "Ugh, her apartment was too cutesie"?

These difficult questions come up repeatedly when you read through the lexicon. Of course, VADER is designed to generalize beyond individual responses and interpretations, and it is not expected to capture the nuances of every single text. At the same time, whenever we use sentiment analysis tools, we want to ask: Does it matter that we're missing out on specific nuances? Is this the best tool for capturing what we're trying to study and understand?

In addition to its lexicon, VADER also calculates sentiment by considering 5 relatively simple rules:

> 1. If there's punctuation, especially exclamation points, the sentiment intensity should be increased (e.g., "Mochi ice cream is bliss" 👍 vs "Mochi ice cream is bliss!!!" 👍👍👍 )

> 2. If there's capitalization, especially all caps, the sentiment intensity should be increased (e.g., "Mochi ice cream is bliss" 👍 vs "Mochi ice cream is BLISS" 👍👍👍 )

> 3. If there are words like "extremely" or "absolutely", the sentiment should be increased ("Mochi ice cream is good" 👍  vs "Mochi ice cream is extremely good" 👍👍👍 )

> 4. If there's a "but" in a sentence, the polarity of the sentiment should shift, and the sentiment that follows the "but" should be prioritized (e.g., "Mochi ice cream is nice" 👍 vs "Mochi ice cream is nice, but it's a little blah" 👎 )

> 5. If there's a negation before an important word, the sentiment polarity should be flipped ("Mochi ice cream is my favorite" 👍 vs "Mochi ice cream is not my favorite" 👎 )


Because VADER uses this lexicon and these simple rules, it works very fast and doesn't require any training or set up, unlike more sophisticated machine learning approaches. The simplicity is both its pro and con.

## Install and Import Libraries/Packages


So let's explore VADER!

To use it, we need to install the [vaderSentiment package](https://github.com/cjhutto/vaderSentiment) with pip.


```python
# !pip install vaderSentiment
```


```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER so we can use it later
sentimentAnalyser = SentimentIntensityAnalyzer()
```

We are also going to import pandas for working with data and set the column width for pandas DataFrames to 400.


```python
import pandas as pd
pd.options.display.max_colwidth = 400
```

## Calculate Sentiment Scores

To calculate sentiment scores for a sentence or paragraph, we can use `sentimentAnalyser.polarity_scores()` and input a string of text.

This method returns a Python dictionary of sentiment scores: how negative the sentence is between 0-1, how neutral the sentence is between 0-1, how positive the sentence is between 0-1, as well as a compound score between -1-1.

Most scholars uses the compound score to represent sentiment in their analyses, and we will use the compound score in this lesson, as well. Let's test it out with some sample sentences! 


```python
sentimentAnalyser.polarity_scores("I like the Marvel movies")
```


```python
sentimentAnalyser.polarity_scores("I don't like the Marvel movies")
```


```python
sentimentAnalyser.polarity_scores("I despise the Marvel movies with every fiber of my being")
```


```python
sentimentAnalyser.polarity_scores("I don't *not* like the Marvel movies")
```

To make the scores more readable, below we make a list of sentences, loop through the list and calculate the polarity scores for each sentence, then print out each part of the sentiment scores dictionary in a nicely formatted f-string (a string that begins with f and allows you to insert Python variables).


```python
# List of sentences
sentences = ["I like the Marvel movies",
             "I don't like the Marvel movies",
             "I despise the Marvel movies with every fiber of my being",
             "I don't *not* live the Marvel movies"]

# Loop through list of sentences
for sentence in sentences:
    # Run VADER on each sentence
    sentiment_scores = sentimentAnalyser.polarity_scores(sentence)
    
    # Print scores for each sentence
    print(f"""'{sentence}' \n
🙁 Negative Sentiment: {sentiment_scores['neg']} \n  
😐 Neutral Sentiment: {sentiment_scores['neu']} \n
😀 Positive Sentiment: {sentiment_scores['pos']} \n
✨ Compound Sentiment: {sentiment_scores['compound']} \n
--- \n""")
```

We can see moments where VADER is both working and not working in the examples above. VADER registers that "I like the Marvel movies" represents the overall most positive sentiment, while "I don't like the Marvel movies" is the most negative sentiment, and "I don't *not* like the Marvel movies" is negative but not quite as negative as a straight up "don't like."

However, VADER does not capture that "I despise the Marvel movies with every fiber of my being" should be the *most* negative sentiment of the bunch. In fact, VADER scores this sentence in the mildly positive range. This example should remind us that VADER typically works best when it is used in aggregate and applied to many examples, such that specific nuances and exceptions like these matter less and come out in the wash.

## Calculating Sentiment Scores: Student Debt Relief related tweets

Let's try using VADER on tweets related to the Biden Administration's student loan forginess program. You can download the dataset [here](https://drive.google.com/drive/u/0/folders/15MC7UJd5Sz0hSKqY2XsFUbFbAM-t065a).

### Load the dataset

We will load the Student Debt Relief JSON file with pandas and drop some of the columns that we don't need.


```python
tweets_df = pd.read_json('student_loan_json.jsonl', orient='split', convert_dates = True,
                       keep_default_dates = ['created_at'])
```


```python
tweets_df.rename(columns={'created_at': 'date',
                          'public_metrics.retweet_count': 'retweets'},
                            inplace=True)
```


```python
tweets_df = tweets_df[['date', 'text', 'retweets']]
```


```python
tweets_df
```

### Calculate Sentiment for All Rows in a Dataframe

To calculate the sentiment for each tweet in the dataframe and add a new column that contains this information, we will create a function that will take in any text and output the compound sentiment score.


```python
def calculate_sentiment(text):
    # Run VADER on the text
    scores = sentimentAnalyser.polarity_scores(text)
    # Extract the compound score
    compound_score = scores['compound']
    # Return compound score
    return compound_score
```

Let's test it out!


```python
calculate_sentiment('I like the Marvel movies')
```

Nice, it works! Now we can apply it to every row in the dataframe with the `.apply()` method. In the same line of code, we are making new column "sentiment_score", where we are outputting our results.


```python
# Apply the function to every row in the "text" column and output the results into a new column "sentiment_score"
tweets_df['sentiment_score'] = tweets_df['text'].apply(calculate_sentiment)
```

Let's sort the DataFrame and examine the top 10 tweets with the highest compound sentiment.


```python
tweets_df.sort_values(by='sentiment_score', ascending=False)[:10]
```

Let's sort the DataFrame and examine the 10 tweets with the lowest compound sentiment.


```python
tweets_df.sort_values(by='sentiment_score', ascending=True)[:10]
```

### Plot Sentiment Over Time

We can plot how the sentiment pf student debt relief tweets fluctuates over time by first converting the date column to a datetime value and then making it the index of the DataFrame, which makes it easier to work with time series data.


```python
tweets_df['date'] = pd.to_datetime(tweets_df['date'])

# Make date the index of the DataFrame
tweets_df = tweets_df.set_index('date')
```


```python
tweets_df.head(2)
```

Then we will group the tweets by month using `.resample()`, a special method for datetime indices, and calculate the average (`.mean()`) compound score for each month. Finally, we will plot these averages.


```python
tweets_df.resample('M')['sentiment_score'].mean().plot(
    title="Student Debt Relief Tweet Sentiment by Month");
```

We can also `.resample()` by day ('D'), week ('W'), or year ('Y').


```python
tweets_df.resample('W')['sentiment_score'].mean().plot(
    title="Student Debt Relief Tweet Sentiment by Week");
```


```python
tweets_df.resample('D')['sentiment_score'].mean().plot(
    title="Student Debt Relief Tweet Sentiment by Day");
```

Looks like there's a dip at the end of December. By using `.loc`, we can also zoom in on particular time periods. Let's take a closer look!


```python
tweets_df.loc["12/1/2022":"1/1/2023"].resample('D')['sentiment_score'].mean().plot(
    title="Student Debt Relief Tweet Sentiment by Day");
```


```python
tweets_df.loc["12/1/2022":"1/1/2023"].sort_values(by='sentiment_score')[:10]
```

## Calculate Sentiment Scores for a State of the Union Address

In this section, we are going to calculate sentiment scores for President Biden's 2023 State of the Union Address. 

First, we need use web scraping tools to collect the transcript from the 2023 State of the Union Address. This White House [URL](https://www.whitehouse.gov/briefing-room/speeches-remarks/2023/02/07/remarks-of-president-joe-biden-state-of-the-union-address-as-prepared-for-delivery/) contains the complete transcript. 

To start, we need to bring in our "requests" library into our Python environment and next we can make our data request using the URL:


```python
import requests
```


```python
response = requests.get("https://www.whitehouse.gov/briefing-room/speeches-remarks/2023/02/07/remarks-of-president-joe-biden-state-of-the-union-address-as-prepared-for-delivery/")
```

Next, we can check to see whether or not the request was successful:


```python
response
```

In order to get the text data from the response we need to apply the .text method, and we can save the results in a new varibale hltm_string. The results from the data request will be in [HTML format](https://www.udacity.com/blog/2021/04/html-for-dummies.html).


```python
html_string = response.text
print(html_string)
```

Let's bring in our [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) Python library to help us clean up and decode this HTML text data:


```python
from bs4 import BeautifulSoup
```

Let's run our html_string variable through the Beautiful Soup object and use the get_text() function to extract the text from the HTML data. Then, let's use the print function to visualize our results:


```python
soup = BeautifulSoup(html_string)
speech = soup.get_text()
print(speech)
```

Let's save our results in a text file:


```python
with open("2023_union.txt","w") as file:
    file.write(speech)
```

Next, let's read in the text file and also replace line breaks with spaces to because there are line breaks in the middle of sentences.


```python
# Read in text file
text = open("2023_union.txt").read()
# Replace line breaks with spaces
text = text.replace('\n', ' ')
```

### Import NLTK

Next we need to break the text into sentences.

An easy way to break text into sentences, or to "tokenize" them into sentences, is to use [NLTK](https://www.nltk.org/), a Python library for text analysis natural language processing.

Let's import nltk and download the model that will help us get sentences.


```python
import nltk
nltk.download('punkt')
```

To break a string into individual sentences, we can use `nltk.sent_tokenize()`


```python
nltk.sent_tokenize(text)
```

To get sentence numbers for each sentence, we can use `enumerate()`.


```python
for number, sentence in enumerate(nltk.sent_tokenize(text)):
    print(number, sentence)
```

### Make DataFrame

For convenience, we can put all of the sentences into a pandas DataFrame. One easy way to make a DataFrame is to first make a list of dictionaries.

Below we loop through the sentences, calculate sentiment scores, and then create a dictionary with the sentence, sentence number, and compound score, which we append to the list `sentence_scores`.


```python
# Break text into sentences
sentences = nltk.sent_tokenize(text)

# Make empty list
sentence_scores = []
# Get each sentence and sentence number, which is what enumerate does
for number, sentence in enumerate(sentences):
    # Use VADER to calculate sentiment
    scores = sentimentAnalyser.polarity_scores(sentence)
    # Make dictionary and append it to the previously empty list
    sentence_scores.append({'sentence': sentence, 'sentence_number': number+1, 'sentiment_score': scores['compound']})
```

To make this list of dictionaries into a DataFrame, we can simply use `pd.DataFrame()`


```python
pd.DataFrame(sentence_scores)
```

Let's examine the 10 most negative sentences.


```python
# Assign DataFrame to variable red_df
speech_df = pd.DataFrame(sentence_scores)

# Sort by the column "sentiment_score" and slice for first 10 values
speech_df.sort_values(by='sentiment_score')[:10]
```

Let's examine the 10 most positive sentences.


```python
# Sort by the column "sentiment_score," this time in descending order, and slice for first 10 values
speech_df.sort_values(by='sentiment_score', ascending=False)[:10]
```

### Make a Sentiment Plot

To create a data visualization of sentiment over the course of the 2023 State of the Union Address we can plot the sentiment scores over story time (aka sentence number).


```python
import plotly.express as px
```


```python
fig = px.line(speech_df, x='sentence_number', y="sentiment_score",
             title= "Sentiment Analysis of 2023 State of the Union Address")
fig.show()
```

We can also get a more generalized view by getting a "rolling average" 5 sentences at a time by using the `.rolling()` method with a specified window and storing the results in a new column "speech_roll":


```python
speech_df['speech_roll'] = speech_df.rolling(5)['sentiment_score'].mean()
```


```python
speech_df[:25]
```


```python
fig = px.line(speech_df, x='sentence_number', y="speech_roll",
             title= "Sentiment Analysis of 2023 State of the Union Address")
fig.show()
```


```python

```
