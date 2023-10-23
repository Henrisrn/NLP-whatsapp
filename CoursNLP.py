import requests
from wordcloud import WordCloud
def wikipedia_page(title):
    '''
    This function returns the raw text of a wikipedia page
    given a wikipedia page title
    '''

    params = {
        'action': 'query',
    'format': 'json', # request json formatted content
    'titles': title, # title of the wikipedia page
    'prop': 'extracts',
    'explaintext': True
    }

    # send a request to the wikipedia api
    response = requests.get(
    'https://fr.wikipedia.org/wiki/',
    params= params
    ).json()

    # Parse the result
    page = next(iter(response['query']['pages'].values()))


    # return the page content
    if 'extract' in page.keys():
        return page['extract']
    else:
        return "Page not found"
# first get the text from the wikipedia page with
# this is the url for Alice in Wonderland
"""result = requests.get('http://www.gutenberg.org/files/11/11-0.txt')
text =result.text"""
print(wikipedia_page("Bernard_Arnault"))
text = wikipedia_page("Bernard_Arnault")

# Instantiate / create a new wordcloud.
wordcloud = WordCloud(
 random_state = 8,
 normalize_plurals = False,
 width = 600,
height= 300,
 max_words = 300,
 stopwords = []
)

# Apply the wordcloud to the text.
wordcloud.generate(text)
# Import matplotlib
import matplotlib.pyplot as plt

# create a figure
fig, ax = plt.subplots(1,1, figsize = (9,6))


# add interpolation = bilinear to smooth things out
plt.imshow(wordcloud, interpolation='bilinear')

# and remove the axis
plt.axis("off")