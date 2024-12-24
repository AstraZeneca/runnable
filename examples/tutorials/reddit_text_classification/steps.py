import logging
import re
from html import unescape

import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nlp = spacy.load("en_core_web_sm")

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class CleanTransformer:
    __uplus_pattern = re.compile("\<[uU]\+(?P<digit>[a-zA-Z0-9]+)\>")
    __markup_link_pattern = re.compile("\[(.*)\]\((.*)\)")

    def predict(self, X, feature_names=[]):
        logger.warning(X)
        f = np.vectorize(CleanTransformer.transform_clean_text)
        X_clean = f(X)
        logger.warning(X_clean)
        return X_clean

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def transform_clean_text(raw_text):
        try:
            decoded = raw_text.encode("ISO-8859-1").decode("utf-8")
        except:  # noqa: E722
            decoded = raw_text.encode("ISO-8859-1").decode("cp1252")
        html_unescaped = unescape(decoded)
        html_unescaped = re.sub(r"\r\n", " ", html_unescaped)
        html_unescaped = re.sub(r"\r\r\n", " ", html_unescaped)
        html_unescaped = re.sub(r"\r", " ", html_unescaped)
        html_unescaped = html_unescaped.replace("&gt;", " > ")
        html_unescaped = html_unescaped.replace("&lt;", " < ")
        html_unescaped = html_unescaped.replace("--", " - ")
        html_unescaped = CleanTransformer.__uplus_pattern.sub(
            " U\g<digit> ", html_unescaped
        )
        html_unescaped = CleanTransformer.__markup_link_pattern.sub(
            " \1 \2 ", html_unescaped
        )
        html_unescaped = html_unescaped.replace("\\", "")
        return html_unescaped


class TokenizeTransformer:
    __symbols = set("!$%^&*()_+|~-=`{}[]:\";'<>?,./-")

    def predict(self, X, feature_names=[]):
        logger.warning(X)
        f = np.vectorize(TokenizeTransformer.transform_to_token, otypes=[object])
        X_tokenized = f(X)
        logger.warning(X_tokenized)
        return X_tokenized

    def fit(self, X, y=None, **fit_params):
        return self

    @staticmethod
    def transform_to_token(text):
        str_text = str(text)
        doc = nlp(str_text, disable=["parser", "tagger", "ner"])
        tokens = []
        for token in doc:
            if token.like_url:
                clean_token = "URL"
            else:
                clean_token = token.lemma_.lower().strip()
                if len(clean_token) < 1 or clean_token in TokenizeTransformer.__symbols:
                    continue
            tokens.append(clean_token)
        return tokens


def extract_text(url: str, encoding: str, features_column: str, labels_column: str):
    df = pd.read_csv(url, encoding=encoding)

    df.to_csv("reddit_text", index=False, header=False)

    x = df[features_column].values
    y = df[labels_column].values

    return x, y


def clean(x: pd.DataFrame):
    clean_text_transformer = CleanTransformer()

    cleaned_x = clean_text_transformer.predict(x)

    return cleaned_x


def tokenize(cleaned_x: pd.DataFrame):
    tokeniser = TokenizeTransformer()

    tokenised_x = tokeniser.predict(cleaned_x)
    return tokenised_x


def tfidf(tokenised_x: pd.DataFrame, max_features: int, ngram_range: int):
    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features,
        preprocessor=lambda x: x,  # We're using cleantext
        tokenizer=lambda x: x,  # We're using spacy
        token_pattern=None,
        ngram_range=(1, ngram_range),
    )

    tfidf_vectorizer.fit(tokenised_x)

    vectorised_x = tfidf_vectorizer.transform(tokenised_x)
    return vectorised_x


def model_fit(vectorised_x: pd.DataFrame, labels: pd.Series, c_param: float):
    lr_model = LogisticRegression(C=c_param, solver="sag")

    lr_model.fit(vectorised_x, labels)

    y_probabilities = lr_model.predict_proba(vectorised_x)

    return y_probabilities, lr_model
