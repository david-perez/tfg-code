import string

import spacy


class SpacyAnalyzer:
    def __init__(self):
        self.nlp = spacy.load('en', disable=['tagger', 'parser', 'ner'])

        def remove_punctuation(doc):
            return [token for token in doc if not any(char in set(string.punctuation) for char in token.lower_)]

        def remove_numerical_tokens(doc):
            return [token for token in doc if not any(char.isdigit() for char in token.lower_)]

        def remove_stop_words(doc):
            # Stop words in spaCy are lowercase.
            return [token for token in doc if not token.lower_ in spacy.lang.en.stop_words.STOP_WORDS]

        def remove_whitespace(doc):
            return [token for token in doc if not token.is_space]

        def remove_one_char_tokens(doc):
            return [token for token in doc if len(token) > 1]

        self.nlp.add_pipe(remove_stop_words, name='remove_stop_words', first=True)
        self.nlp.add_pipe(remove_one_char_tokens, name='remove_one_word_tokens', first=True)
        self.nlp.add_pipe(remove_numerical_tokens, name='remove_numerical_tokens', first=True)
        self.nlp.add_pipe(remove_whitespace, name='remove_whitespace', first=True)
        self.nlp.add_pipe(remove_punctuation, name='remove_punctuation', first=True)

    def analyze(self, text):
        doc = self.nlp(text)
        return [token.lower_ for token in doc]
