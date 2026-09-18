"""Loading reviews from the retail dataset into Polars and shaping them for classification."""

from .review_loader import content_key, load_reviews, product_catalogue
from .sentence_splitter import split_sentences

__all__ = ["content_key", "load_reviews", "product_catalogue", "split_sentences"]
