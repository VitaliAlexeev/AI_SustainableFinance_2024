#############################################
# Keep this
#############################################
import sys
import subprocess

def install_packages(element):
    try:
        print(f"Installing package \"{element}\"...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", element]) 
        print(f"DONE: Package {element} is up to date.")
    except:
        print(f"ERROR: Unable to download \"{element}\" package, please check your internet connection or package name spelling and try again.")
    return None

try:
    import pylibcheck
    print(f"Package pylibcheck is installed and loaded.")
except:
    print(f"Installing pylibcheck... Package pylibcheck is installed.")
    install_packages("pylibcheck")
    import pylibcheck



#############################################
# Check if libraries are installed
#############################################
# Add additional libraries to check in the list below:
packages_list = ["numpy",
                "Wikipedia-API",
                "pandas",
                "matplotlib",
                "seaborn",
                "contractions",
                "nltk",
                "wordcloud",
                "plotly"]    

def check_packages(packages_list):
    for element in packages_list:
        if pylibcheck.checkPackage(element):
            print(f"OK: Package {element} is installed.")
        else:
            install_packages(element)
    return None

check_packages(packages_list)

#############################################
# Load libraries
#############################################
import wikipediaapi

import numpy as np
from PIL import Image

import re
import string
import contractions
import itertools

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from wordcloud import WordCloud
import matplotlib.pyplot as plt

import plotly.graph_objs as go
from plotly.offline import iplot

# When editing a module, and not wanting to restatrt kernel every time use:
# import importlib
# importlib.reload(bc)
# import utsbootcamp as bc


#############################################
# Functions
#############################################
def download_wikipedia_text(topic, 
                            language='en', 
                            user_agent='MyWikiApp/1.0', 
                            summary=False, 
                            include_url=False, 
                            save_to_file=False, 
                            file_name=None, 
                            include_sections=False, 
                            include_links=False, 
                            include_categories=False, 
                            include_languages=False):
    """
    Downloads text from a Wikipedia page on a specified topic with additional options.

    Parameters:
    - topic (str): The title of the Wikipedia page to download.
    - language (str, optional): The language of the Wikipedia page (default is 'en' for English).
    - user_agent (str, optional): The user agent to use for the Wikipedia API request (default is 'MyWikiApp/1.0').
    - summary (bool, optional): If True, return a summary of the page instead of the full text (default is False).
    - include_url (bool, optional): If True, include the URL of the Wikipedia page in the output (default is False).
    - save_to_file (bool, optional): If True, save the text or summary to a file (default is False).
    - file_name (str, optional): The name of the file to save the content to. If not specified, a file name is generated based on the topic.
    - include_sections (bool, optional): If True, include the sections of the Wikipedia page in the output (default is False).
    - include_links (bool, optional): If True, include the links from the Wikipedia page in the output (default is False).
    - include_categories (bool, optional): If True, include the categories of the Wikipedia page in the output (default is False).
    - include_languages (bool, optional): If True, include the available languages for the Wikipedia page in the output (default is False).

    Returns:
    - str: The text or summary of the Wikipedia page with additional information as specified, or an error message if the page does not exist. If 'save_to_file' is True, returns a message indicating the file where the content is saved.

    Example usage:
    >>> topic = "Machine learning"
    >>> options = {
    ...     'summary': True,
    ...     'include_url': True,
    ...     'include_sections': True,
    ...     'include_links': True,
    ...     'include_categories': True,
    ...     'include_languages': True,
    ...     'save_to_file': True,
    ...     'file_name': 'machine_learning_info.txt'
    ... }
    >>> text = download_wikipedia_text(topic, **options)
    >>> print(text)
    
    The ** operator is used to unpack a dictionary into keyword arguments when calling a function. 
    When you see **options in the function call, it means that the options dictionary is being unpacked and 
    its key-value pairs are passed as keyword arguments to the function.
    
    """
    # Create a Wikipedia API instance with the specified user agent
    wiki_wiki = wikipediaapi.Wikipedia(language=language, user_agent=user_agent)

    # Get the page for the specified topic
    page = wiki_wiki.page(topic)

    # Check if the page exists
    if page.exists():
        content = ""

        # Add summary or full text
        content += page.summary if summary else page.text

        # Add sections
        if include_sections:
            content += "\n\nSections:\n" + "\n".join([section.title for section in page.sections])

        # Add links
        if include_links:
            content += "\n\nLinks:\n" + "\n".join([link for link in page.links])

        # Add categories
        if include_categories:
            content += "\n\nCategories:\n" + "\n".join([category for category in page.categories])

        # Add languages
        if include_languages:
            content += "\n\nLanguages:\n" + "\n".join([lang for lang in page.langlinks])

        # Add URL
        if include_url:
            content += '\n\nURL: ' + page.fullurl

        # Save to file
        if save_to_file:
            file_name = file_name or f"{topic.replace(' ', '_')}.txt"
            with open(file_name, 'w', encoding='utf-8') as file:
                file.write(content)
            return f"Content saved to file: {file_name}"

        # Return content
        return content
    else:
        # Return an error message if the page doesn't exist
        return f"The page for the topic '{topic}' does not exist on Wikipedia."
		
def find_text_elements(text,
                       pattern=r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}|\$[^$]*\$|<[^>]*>',
                       remove=False):
    """
    Finds or removes specific text elements from the input text based on a regular expression pattern.

    Parameters:
    - text (str):              The input text from which to find or remove elements.
    - pattern (str, optional): The regular expression pattern used to identify the text elements. 
                               The default pattern matches text between round brackets (parentheses), 
                               square brackets, curly brackets, $ signs for LaTeX expressions, 
                               and < and > signs for HTML tags, without nesting.
                               Default pattern: r'\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}|\$[^$]*\$|<[^>]*>'
                               where:
                               \( [^()]* \)    : Matches text between round brackets (parentheses) without nesting.
                               \[ [^\[\]]* \]  : Matches text between square brackets without nesting.
                               \{ [^{}]* \}    : Matches text between curly brackets without nesting.
                               \$ [^$]* \$     : Matches text between $ signs for LaTeX expressions.
                               < [^>]* >       : Matches text between < and > signs for HTML tags.
    - remove (bool, optional): If True, the identified text elements are removed from the input text. 
                               If False, the identified text elements are returned as a list. 
                               Default value is False.

    Returns:
    - If remove is False, returns a list of matches found in the input text based on the pattern.
    - If remove is True, returns the input text with the identified elements removed.

    Examples:
    >>> text = "This is a sample text with (parentheses), [brackets], {curly brackets}, $LaTeX$ expression, and <html> tags."
    >>> find_text_elements(text)
    ['(parentheses)', '[brackets]', '{curly brackets}', '$LaTeX$', '<html>']
    
    >>> find_text_elements(text, remove=True)
    'This is a sample text with , , , , and .'

    Note:
    - The default pattern is designed to match specific text elements without nesting. 
      If you need to handle nested structures, you will need to modify the pattern accordingly.
    - The regular expression patterns can be customized to match different text elements 
      as per the requirements of your application.
    """    
   
    if remove:
        text = re.sub(pattern, '', text)
        return text
    else:
        matches = re.findall(pattern, text)
        return matches

def remove_text_inside_brackets(text, brackets='''{}()[]'''):
    count = [0] * (len(brackets) // 2) # count open/close brackets
    saved_chars = []
    for character in text:
        for i, b in enumerate(brackets):
            if character == b: # found bracket
                kind, is_close = divmod(i, 2)
                count[kind] += (-1)**is_close # `+1`: open, `-1`: close
                if count[kind] < 0: # unbalanced bracket
                    count[kind] = 0  # keep it
                else:  # found bracket to remove
                    break
        else: # character is not a [balanced] bracket
            if not any(count): # outside brackets
                saved_chars.append(character)
    return ''.join(saved_chars)


def preprocess_text(text, 
                    remove_text_in_brackets=True,
                    remove_url=True,
                    remove_html_tags=True,
                    to_lower=True, 
                    expand_contractions=True,
                    remove_punctuation=True, 
                    remove_digits=True, 
                    remove_stopwords=True, 
                    lemmatize=True, 
                    stem=False, 
                    custom_stopwords=None,
                    custom_brackets=None):
    """
    Preprocesses and cleans text with various options.
    
    This function provides options for converting text to lowercase, removing punctuation, removing digits, 
    removing stopwords, applying lemmatization, applying stemming, and specifying custom stopwords. 
    You can adjust these options according to your needs by setting the corresponding parameters to True or False. 
    For example, if you want to keep the punctuation, you can call the function with 'remove_punctuation=False'.

    Parameters:
    - text (str): The text to be preprocessed.
    - to_lower (bool, optional): Convert text to lowercase (default is True).
    - remove_punctuation (bool, optional): Remove punctuation from text (default is True).
    - remove_digits (bool, optional): Remove digits from text (default is True).
    - remove_stopwords (bool, optional): Remove stopwords from text (default is True).
    - lemmatize (bool, optional): Apply lemmatization to words (default is True).
    - stem (bool, optional): Apply stemming to words (default is False).
    - custom_stopwords (list, optional): A list of custom stopwords to remove (default is None).

    Returns:
    - str: The preprocessed and cleaned text.

    Example usage:
    >>> text = "The quick brown fox jumps over the lazy dog."
    >>> cleaned_text = preprocess_text(text)
    >>> print(cleaned_text)
    """
    
    # Remove brackets 
    if remove_text_in_brackets:
        if custom_brackets:
            text=remove_text_inside_brackets(text, brackets=custom_brackets)
        else:
            text=remove_text_inside_brackets(text)
            
    
    # Remove URLs
    if remove_url:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
    # Remove HTML tags
    if remove_html_tags:
        text = re.sub(r'<.*?>', '', text)
        
    # Expanding Contractions
    if expand_contractions:
        text = contractions.fix(text)
        
    # Convert text to lowercase
    if to_lower:
        text = text.lower()

    # Remove punctuation
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove digits
    if remove_digits:
        text = re.sub(r'\d+', '', text)

    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        if custom_stopwords:
            stop_words.update(custom_stopwords)
        tokens = [word for word in tokens if word not in stop_words]

    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Apply lemmatization
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Apply stemming
    if stem:
        tokens = [stemmer.stem(word) for word in tokens]

    # Rejoin tokens into a single string
    cleaned_text = ' '.join(tokens)

    return cleaned_text
	
def plot_wordcloud(text,
                   width=3000, height=2000, 
                   background_color='salmon', 
                   colormap='Pastel1', 
                   collocations=False, 
                   stopwords=None,
                   figsize=(18, 14),
                   mask=None,
                   min_word_length=0,
                   include_numbers=False):
    """
    Generates and plots a word cloud from the input text.

    Parameters:
    - text (str): The input text for generating the word cloud.
    - width (int, optional): Width of the word cloud image. Default is 3000.
    - height (int, optional): Height of the word cloud image. Default is 2000.
    - background_color (str, optional): Background color of the word cloud image. Default is 'salmon'.
    - colormap (str, optional): Colormap for coloring the words. Default is 'Pastel1'.
    - collocations (bool, optional): Whether to include collocations (bigrams) in the word cloud. Default is False.
    - stopwords (set, optional): Set of stopwords to exclude from the word cloud. Default is STOPWORDS from wordcloud.
    - figsize (tuple, optional): Size of the figure for plotting the word cloud. Default is (40, 30).
    - mask (string, optional): Path and file name to masking image file
    
    Returns:
    - None
    
    Example usage:
    text = "Python is a great programming language for data analysis and visualization. Python is popular for data science."
    plot_wordcloud(text,stopwords=['is','a'])

    """
    if mask:
        # Import image to np.array
        mask = np.array(Image.open(mask))

    # Generate word cloud
    wordcloud = WordCloud(width=width, height=height, 
                          background_color=background_color, 
                          colormap=colormap, 
                          collocations=collocations, 
                          stopwords=stopwords,
                          mask=mask,
                          min_word_length=min_word_length,
                          include_numbers=include_numbers).generate(text)

    # Plot word cloud
    plt.figure(figsize=figsize)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def print_colored_text(text_data):
    """
    Prints elements of a list in different colors.

    This function takes a single argument, 'text_data', which can be either a list of strings or a list of lists of strings.
    It prints each element in 'text_data' in a different color, cycling through a predefined set of colors. If 'text_data' is
    a list of lists, each sublist is printed in a single color.

    Parameters:
    - text_data (list): A list of strings or a list of lists of strings to be printed in different colors.

    Usage:
    - print_colored_text(["string1", "string2", "string3"])
    - print_colored_text([["word1", "word2"], ["word3", "word4"], ["word5", "word6"]])
    
    Or:
    data = ["string1", "string2", "string3", "string4"]
    data_words = [["word1", "word2"], ["word3", "word4"], ["word5", "word6"], ["word7", "word8"]]
    print_colored_text(data)
    print_colored_text(data_words)
    """
   
    # Define colors
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'reset': '\033[0m'
    }

    # Create a cycle of colors
    color_cycle = itertools.cycle(colors.values())

    # Check if 'text_data' is a list of lists of strings
    if all(isinstance(item, list) for item in text_data):
        # 'text_data' is a list of lists of strings
        print("Data Type: a list of lists of strings:")
        for sublist in text_data[:4]:
            color = next(color_cycle)
            print(f"{color}[", end="")
            for word in sublist:
                print(f"{word} ", end="")
            print(f"]{colors['reset']}")
    elif all(isinstance(item, str) for item in text_data):
        # 'text_data' is a list of strings
        print("Data Type: a list of strings")
        for item in text_data[:4]:
            color = next(color_cycle)
            print(f"{color}{item}{colors['reset']}")
    else:
        print("Invalid input: 'text_data' must be a list of strings or a list of lists of strings.")
    print('\n')