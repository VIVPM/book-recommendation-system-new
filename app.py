from books_recommender.exception.exception_handler import AppException
from books_recommender.logger.log import logging
import sys

try:
    a = 3/0
except Exception as e:
    logging.info("Divide by zero")
    raise AppException(e, sys)