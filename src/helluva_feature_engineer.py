import editdistance
import numpy as np
import pandas
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

from src import load_data

# This file takes in a CSV of pairs with all of the original data and
# outputs a completely numerical dataset that can be used for training
# and testing.

TITLE = 'title'
CATEGORY = 'category'
MODEL = 'modelno'
PRICE = 'price'

UNENTERED_PATH = "ENTER_PATH_HERE"
# ENTER YOUR OWN PATH HERE!
PATH_TO_GOOGLE_MODEL = UNENTERED_PATH
if PATH_TO_GOOGLE_MODEL == UNENTERED_PATH:
	print("HEY! You need to input the path to the google word2vec model"
		  "on line 16! #Error")
	import sys
	sys.exit(1)

print("LOADING MODEL")
GOOGLE_MODEL = KeyedVectors.load_word2vec_format(PATH_TO_GOOGLE_MODEL, binary=True)
WORDS_TO_IGNORE = stopwords.words("english")
print("LOADED MODEL")


def engineer_features(all_pairs_df):
	new_df = pandas.DataFrame(columns=['id1', 'id2', 'price_diff', 'model_match', 'category_match', 'description_similarity'])

	features_engineered = 0
	for idx, row in all_pairs_df.iterrows():
		id_l = row['id_l']
		id_r = row['id_r']

		price_l = row[PRICE + '_l']
		price_r = row[PRICE + '_r']
		price_diff = np.nan if price_l is None or price_r is None \
			else price_l - price_r

		model_l = row[MODEL + '_l']
		model_r = row[MODEL + '_r']
		mm = np.nan if model_l is None or model_r is None \
			else find_longest_matching_string(str(model_l), str(model_r))

		category_l = row[CATEGORY + '_l']
		category_r = row[CATEGORY + '_r']
		cm = np.nan if category_l is None or category_r is None \
			else find_longest_matching_string(str(category_l), str(category_r))

		brand_l = row['brand_l']
		brand_r = row['brand_r']
		bm = np.nan if brand_l is None or brand_r is None \
			else find_longest_matching_string(str(brand_l), str(brand_r))

		title_vec_l = row[TITLE + '_l']
		title_vec_r = row[TITLE + '_r']
		td = np.nan if title_vec_l is None or title_vec_r is None \
			else get_phrase_score(str(title_vec_l), str(title_vec_r))

		new_df = new_df.append(
			{
				'id1': id_l,
				'id2': id_r,
				'price_diff': price_diff,
				'model_match': mm,
				'category_match': cm,
				'brand_match': bm,
				'description_similarity': td
			},
			ignore_index=True)

		features_engineered += 1
		if features_engineered % 100 == 0:
			print(features_engineered)
		if features_engineered == 100:
			new_df.to_csv(load_data.TRAIN_PARTIAL_PATH)

	new_df.to_csv(load_data.TRAIN_PATH_BRAND_FUZZY_UPDATE)


def get_phrase_score(phrase_one, phrase_two):
	pv_one = PhraseVector(phrase_one)
	pv_two = PhraseVector(phrase_two)
	return pv_one.get_similarity(pv_two)


def find_longest_matching_string(string_a, string_b):
	ed = editdistance.eval(string_a, string_b)
	max_len = max(len(string_a), len(string_b))
	return (max_len - ed) / max_len


class PhraseVector:
	def __init__(self, phrase):
		self.vector_set = self.phrase_to_vec_list(phrase)

	def phrase_to_vec_list(self, phrase):
		phrase = phrase.lower()
		words = [w for w in phrase.split() if w not in WORDS_TO_IGNORE]
		vector_list = []
		for w in words:
			try:
				word_vec = GOOGLE_MODEL[w]
				vector_list.append(word_vec)
			except:
				pass
		return vector_list

	def get_similarity(self, other):
		best_matches = []
		for vec in self.vector_set:
			best_match = -2 # worse than cos(180)
			for other_vec in other.vector_set:
				match = self.cosine_similarity(vec, other_vec)
				if match > best_match:
					best_match = match
			best_matches.append(best_match)
		return np.mean(best_matches)

	def cosine_similarity(self, pv1, pv2):
		score = np.dot(pv1, pv2) \
				/ (np.linalg.norm(pv1) * np.linalg.norm(pv2))
		try:
			if np.isnan(score):
				score = 0
		except:
			score = 0
		return score


if __name__ == '__main__':
	engineer_features(pandas.read_csv(load_data.ALL_PAIRS_BRAND_FUZZY_PATH))