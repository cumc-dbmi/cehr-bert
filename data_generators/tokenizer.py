from typing import Optional, Sequence, Union
from dask.dataframe import Series as dd_series
from pandas import Series as df_series
from tensorflow.python.keras.preprocessing.text import Tokenizer

BERT_SPECIAL_TOKENS = ['[MASK]', '[UNUSED]', '[CLS']


class ConceptTokenizer:
    unused_token = ['[UNUSED]']
    mask_token = ['[MASK]']
    cls_token = ['[CLS]']

    def __init__(self, special_tokens: Optional[Sequence[str]] = None, oov_token='0'):
        self.special_tokens = special_tokens
        self.tokenizer = Tokenizer(oov_token=oov_token, filters='', lower=False)

    def fit_on_concept_sequences(self, concept_sequences: Union[df_series, dd_series]):

        if isinstance(concept_sequences, df_series):
            self.tokenizer.fit_on_texts(concept_sequences.apply(
                lambda concept_ids: concept_ids.tolist()))
        else:
            self.tokenizer.fit_on_texts(
                concept_sequences.apply(lambda s: s.tolist(), meta='iterable'))

        self.tokenizer.fit_on_texts(self.mask_token)
        self.tokenizer.fit_on_texts(self.unused_token)
        self.tokenizer.fit_on_texts(self.cls_token)
        if self.special_tokens is not None:
            self.tokenizer.fit_on_texts(self.special_tokens)

    def encode(self, concept_sequences, is_generator=False):
        return self.tokenizer.texts_to_sequences_generator(
            concept_sequences) if is_generator else self.tokenizer.texts_to_sequences(
            concept_sequences)

    def decode(self, concept_sequence_token_ids):
        return self.tokenizer.sequences_to_texts(concept_sequence_token_ids)

    def get_all_token_indexes(self):
        all_keys = set(self.tokenizer.index_word.keys())

        if self.tokenizer.oov_token is not None:
            all_keys.remove(self.tokenizer.word_index[self.tokenizer.oov_token])

        if self.special_tokens is not None:
            excluded = set(
                [self.tokenizer.word_index[special_token] for special_token in self.special_tokens])
            all_keys = all_keys - excluded
        return all_keys

    def get_first_token_index(self):
        return min(self.get_all_token_indexes())

    def get_last_token_index(self):
        return max(self.get_all_token_indexes())

    def get_vocab_size(self):
        # + 1 because oov_token takes the index 0
        return len(self.tokenizer.index_word) + 1

    def get_unused_token_id(self):
        unused_token_id = self.encode(self.unused_token)
        while isinstance(unused_token_id, list):
            unused_token_id = unused_token_id[0]
        return unused_token_id

    def get_mask_token_id(self):
        mask_token_id = self.encode(self.mask_token)
        while isinstance(mask_token_id, list):
            mask_token_id = mask_token_id[0]
        return mask_token_id

    def get_cls_token_id(self):
        cls_token_id = self.encode(self.cls_token)
        while isinstance(cls_token_id, list):
            cls_token_id = cls_token_id[0]
        return cls_token_id
