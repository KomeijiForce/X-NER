from nltk.corpus import stopwords
from transformers import AutoTokenizer

stops = stopwords.words('english')
stops = stops + [".", ",", "?", "!", "'", '"', "(", ")", "-", "--", "_", "[", "]", "{", "}", "#", "@", "*"]
nums = list(map(str, range(10)))
words2sent = AutoTokenizer.from_pretrained("bert-large-cased").convert_tokens_to_string