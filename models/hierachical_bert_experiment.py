#!/usr/bin/env python
# coding: utf-8
# %%
import tensorflow as tf

# %%
from models.custom_layers import *
from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)
import numpy as np


# %%
def transformer_hierarchical_bert_model(num_of_visits,
                                        num_of_concepts,
                                        concept_vocab_size,
                                        visit_vocab_size,
                                        embedding_size,
                                        depth: int,
                                        num_heads: int,
                                        transformer_dropout: float = 0.1,
                                        embedding_dropout: float = 0.6,
                                        l2_reg_penalty: float = 1e-4,
                                        time_embeddings_size: int = 16):
    
    pat_seq = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32', name='pat_seq')
    pat_seq_age = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32', name='pat_seq_age')
    pat_seq_time = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32', name='pat_seq_time')
    pat_mask = tf.keras.layers.Input(shape=(num_of_visits, num_of_concepts,), dtype='int32', name='pat_mask')

    visit_time_delta_att = tf.keras.layers.Input(shape=(num_of_visits-1,), dtype='int32',
                                                 name='visit_time_delta_att')
    visit_mask = tf.keras.layers.Input(shape=(num_of_visits,), dtype='int32', name='visit_mask')

    default_inputs = [pat_seq, pat_seq_age, pat_seq_time, 
                      pat_mask, visit_time_delta_att, visit_mask]

    pat_concept_mask = tf.reshape(pat_mask, (-1, num_of_concepts))[:, tf.newaxis, tf.newaxis, :]

    visit_mask_with_att = tf.reshape(tf.stack([visit_mask, visit_mask], axis=2),
                                     (-1, num_of_visits * 2))[:, 1:]

    visit_concept_mask = visit_mask_with_att[:, tf.newaxis, tf.newaxis, :]

    # output the embedding_matrix:
    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size, 
        embedding_size,
        name='bpe_embeddings',
        embeddings_regularizer=l2_regularizer
    )

    # # define the time embedding layer for absolute time stamps (since 1970)
    time_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                              name='time_embedding_layer')
    # define the age embedding layer for the age w.r.t the medical record
    age_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                             name='age_embedding_layer')

    temporal_transformation_layer = tf.keras.layers.Dense(embeddinig_size,
                                                          activation='tanh',
                                                          name='temporal_transformation')

    pt_seq_concept_embeddings, embedding_matrix = concept_embedding_layer(pat_seq)
    pt_seq_age_embeddings = age_embedding_layer(pat_seq_age)
    pt_seq_time_embeddings = time_embedding_layer(pat_seq_time)

    # dense layer for rescale the patient sequence embeddings back to the original size
    temporal_concept_embeddings = temporal_transformation_layer(
        tf.concat([pt_seq_concept_embeddings, pt_seq_age_embeddings, pt_seq_time_embeddings],
                  axis=-1, name='concat_for_encoder'))

    temporal_concept_embeddings = tf.reshape(temporal_concept_embeddings,
                                             (-1, num_of_concepts, embeddinig_size))

    # The first bert applied at the visit level
    concept_encoder = Encoder(name='concept_encoder',
                              num_layers=depth,
                              d_model=embeddinig_size,
                              num_heads=num_heads,
                              dropout_rate=transformer_dropout)

    contextualized_concept_embeddings, _ = concept_encoder(
        temporal_concept_embeddings,
        pat_concept_mask
    )

    contextualized_concept_embeddings = tf.reshape(
        contextualized_concept_embeddings,
        shape=(-1, num_of_visits, num_of_concepts, embeddinig_size)
    )

    # Slice out the first contextualized embedding of each visit
    visit_embeddings = contextualized_concept_embeddings[:, :, 0]

    # Reshape the data in visit view back to patient view: (batch, sequence, embedding_size)
    contextualized_concept_embeddings = tf.reshape(
        contextualized_concept_embeddings,
        shape=(-1, num_of_visits * num_of_concepts, embeddinig_size)
    )

    # Insert the att embeddings between the visit embeddings using the following trick
    identity = tf.constant(
        np.insert(
            np.identity(num_visit),
            obj=range(1, num_visit),
            values=0,
            axis=1
        ),
        dtype=tf.float32
    )
    expanded_visit_embeddings = tf.transpose(
        tf.transpose(visit_embeddings, perm=[0, 2, 1]) @ identity,
        perm=[0, 2, 1]
    )

    # Look up the embeddings for the att tokens
    att_embeddings, _ = concept_embedding_layer(visit_time_delta_att)
    # Create the inverse "identity" matrix for inserting att embeddings
    identity_inverse = tf.constant(
        np.insert(
            np.identity(num_of_visits - 1),
            obj=range(0, num_of_visits),
            values=0,
            axis=1),
        dtype=tf.float32)

    expanded_att_embeddings = tf.transpose(
        tf.transpose(att_embeddings, perm=[0, 2, 1]) @ identity_inverse,
        perm=[0, 2, 1]
    )

    # Insert the att embeddings between visit embedidngs
    augmented_visit_embeddings = expanded_visit_embeddings + expanded_att_embeddings

    # Second bert applied at the patient level to the visit embeddings
    visit_encoder = Encoder(name='visit_encoder',
                            num_layers=depth,
                            d_model=embeddinig_size,
                            num_heads=num_heads,
                            dropout_rate=transformer_dropout)
    # Feed augmented visit embeddings into encoders to get contextualized visit embeddings
    contextualized_visit_embeddings, _ = visit_encoder(
        augmented_visit_embeddings,
        visit_concept_mask
    )

    # Use a multi head attention layer to generate the global concept embeddings by attending to
    # the visit embeddings
    multi_head_attention_layer = MultiHeadAttention(embeddinig_size, num_heads)
    global_concept_embeddings, _ = multi_head_attention_layer(
        contextualized_visit_embeddings,
        contextualized_visit_embeddings,
        contextualized_concept_embeddings,
        visit_concept_mask,
        None)

    concept_output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    contextualized_visit_embeddings_without_att = identity @ contextualized_visit_embeddings

    visit_prediction_dense = tf.keras.layers.Dense(visit_vocab_size)

    concept_softmax_layer = tf.keras.layers.Softmax(name='concept_predictions')
    visit_softmax_layer = tf.keras.layers.Softmax(name='visit_predictions')

    concept_predictions = concept_softmax_layer(
        concept_output_layer([global_concept_embeddings, embedding_matrix])
    )

    visit_predictions = visit_softmax_layer(
        visit_prediction_dense(contextualized_visit_embeddings_without_att)
    )

    hierarchical_bert = tf.keras.Model(
        inputs=default_inputs,
        outputs=[concept_predictions, visit_predictions])

    return hierarchical_bert


# %%
concepts = tf.random.uniform((1, 1000), dtype=tf.int32, minval=1, maxval=1000)
time_stamps = tf.sort(tf.random.uniform((1, 1000), dtype=tf.int32, maxval=1000))
ages = tf.sort(tf.random.uniform((1, 1000), dtype=tf.int32, minval=18, maxval=80))
mask = tf.sort(tf.random.uniform((1, 1000), dtype=tf.int32, maxval=2))

visit_time_stamps = tf.sort(tf.random.uniform((1, 20), dtype=tf.int32, maxval=1000))
visit_seq_time_delta = tf.sort(tf.random.uniform((1, 19), dtype=tf.int32, maxval=1000))
visit_mask = tf.sort(tf.random.uniform((1, 20), dtype=tf.int32, maxval=2))

# %%
num_concept_per_v = 50
num_visit = 10
num_seq = num_concept_per_v * num_visit

concept_vocab_size = 40000
visit_vocab_size = 10

embeddinig_size = 128
time_embeddings_size = 16
depth = 16
num_heads = 8
transformer_dropout: float = 0.1
embedding_dropout: float = 0.6
l2_regularizer = tf.keras.regularizers.l2(1e-4)

# %%
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = transformer_hierarchical_bert_model(num_visit,
                                            num_concept_per_v,
                                            tokenizer.get_vocab_size(),
                                            visit_tokenizer.get_vocab_size(),
                                            embeddinig_size,
                                            depth,
                                            num_heads)

# %%
model.summary()

# %%
optimizer = tf.optimizers.Adam(
    learning_rate=2e-4, beta_1=0.9, beta_2=0.999)

losses = {
    'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(0.1),
    'visit_predictions': MaskedPenalizedSparseCategoricalCrossentropy(0.1)
}

model.compile(optimizer, loss=losses, metrics={'concept_predictions': masked_perplexity})

# %%
n_of_data_points = 2560

pat_seq_input = tf.random.uniform((n_of_data_points, num_visit, num_concept_per_v), maxval=100, dtype=tf.int32)
pat_seq_age_input = tf.sort(tf.random.uniform((n_of_data_points, num_visit, num_concept_per_v), minval=18, maxval=100, dtype=tf.int32))
pat_seq_time_input = tf.sort(tf.random.uniform((n_of_data_points, num_visit, num_concept_per_v), maxval=1000, dtype=tf.int32))
pat_mask_input = tf.sort(tf.random.uniform((n_of_data_points, num_visit, num_concept_per_v), maxval=2, dtype=tf.int32))

visit_time_delta_att_input = tf.sort(tf.random.uniform((n_of_data_points, num_visit - 1), maxval=20, dtype=tf.int32))
visit_mask_input = tf.sort(tf.random.uniform((n_of_data_points, num_visit), maxval=2, dtype=tf.int32))
inputs = {
    'pat_seq': pat_seq_input,
    'pat_seq_age': pat_seq_age_input,
    'pat_seq_time': pat_seq_time_input,
    'pat_mask': pat_mask_input,
    'visit_time_delta_att': visit_time_delta_att_input,
    'visit_mask': visit_mask_input
}

concepts_types = tf.sort(tf.random.uniform((n_of_data_points, num_seq), maxval=10, dtype=tf.int32))
output_mask = tf.sort(tf.random.uniform((n_of_data_points, num_seq), maxval=2, dtype=tf.int32))

visit_types = tf.sort(tf.random.uniform((n_of_data_points, num_visit), maxval=10, dtype=tf.int32))
output_visit_mask = tf.sort(tf.random.uniform((n_of_data_points, num_visit), maxval=2, dtype=tf.int32))

output_dict = {
    'concept_predictions': np.stack([concepts_types, output_mask], axis=-1),
    'visit_predictions': np.stack([visit_types, output_visit_mask], axis=-1)
}
# %%
dataset = tf.data.Dataset.from_tensor_slices((inputs, output_dict)).cache().batch(8)

# %%
# for x, y in dataset:
#     print(model(x))
#     break

# %%
model.fit(dataset)

# %%
import os
import pandas as pd
from data_generators.data_generator_base import HierarchicalBertDataGenerator
from utils.model_utils import *

# %%
output_folder = '/data/research_ops/omops/omop_2020q2/hierarchical_bert'
patient_sequence = pd.read_parquet(os.path.join(output_folder, 'patient_sequence'))
tokenizer_path = os.path.join(output_folder, 'concept_tokenizer.pickle')
visit_tokenizer_path = os.path.join(output_folder, 'visit_tokenizer.pickle')

# %%
patient_sequence['patient_concept_ids'] = patient_sequence.concept_ids.apply(lambda visit_concepts: np.hstack(visit_concepts))

# %%
from data_generators.data_generator_base import *
class HierarchicalBertDataGenerator(AbstractDataGeneratorBase):

    def __init__(self,
                 concept_tokenizer: ConceptTokenizer,
                 max_num_of_visits: int,
                 max_num_of_concepts: int,
                 sliding_window: int = 10,
                 *args,
                 **kwargs):
        
        super(HierarchicalBertDataGenerator, self).__init__(concept_tokenizer=concept_tokenizer,
                                                            max_num_of_visits=max_num_of_visits,
                                                            max_num_of_concepts=max_num_of_concepts,
                                                            *args, **kwargs)
        self._concept_tokenizer = concept_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts
        self._sliding_window = sliding_window

    def _get_learning_objective_classes(self):
        return [HierarchicalMaskedLanguageModelLearningObjective]

    def _calculate_step(self, num_of_visits):
        """
        Calculate the number of steps used for the sliding window strategy
        :param num_of_visits:
        :return:
        """
        if num_of_visits <= self._max_num_of_visits:
            return 1
        else:
            return math.ceil((num_of_visits - self._max_num_of_visits) / self._sliding_window) + 1
        
    def _create_iterator(self):
        """
        Create an iterator that will iterate forever
        :return:
        """
        while True:
            for row in self._training_data.itertuples():
#                 for step in range(self._calculate_step(row.num_of_visits)):
# #                     end_index = row.num_of_visits - step * self._sliding_window
# #                     start_index = max(end_index - self._max_num_of_visits, 0)
#                     start_index = step * self._sliding_window
#                     end_index = step * self._sliding_window + self._max_num_of_visits

#                     if end_index > row.num_of_visits:
#                         start_index = row.num_of_visits - self._max_num_of_visits
#                         end_index = row.num_of_visits
                        
                yield RowSlicer(row, 0, self._max_num_of_visits)

    def estimate_data_size(self):
        return self._training_data.num_of_visits.apply(self._calculate_step).sum()


# %%
class HierarchicalMaskedLanguageModelLearningObjective(LearningObjective):
    required_columns = ['concept_ids', 'dates',
                        'visit_segments', 'ages',
                        'visit_dates', 'visit_masks',
                        'visit_concept_ids', 'time_interval_atts']

    def __init__(self, concept_tokenizer: ConceptTokenizer,
                 max_num_of_visits: int,
                 max_num_of_concepts: int,
                 is_training: bool):
        self._concept_tokenizer = concept_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts
        self._is_training = is_training

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'pat_seq': int32,
            'pat_seq_age': int32,
            'pat_seq_time': int32,
            'pat_mask': int32,
            'visit_segment': int32,
            'visit_time_delta_att': int32,
            'visit_mask': int32
        }
        output_dict_schema = {'concept_predictions': int32, 'visit_predictions': int32}
        return input_dict_schema, output_dict_schema

    def _pad(self, x, padded_token):
        return pad_sequences(np.asarray(x), maxlen=self._max_num_of_concepts, padding='post',
                             value=padded_token, dtype='int32')

    def _concept_mask(self, concept_ids):
        return list(map(lambda c: (c == self._concept_tokenizer.get_unused_token_id()).astype(int),
                        concept_ids))

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):

        (
            output_concept_masks, masked_concepts, concepts, dates, ages,
            visit_concept_ids, visit_segments, visit_dates, visit_masks, time_interval_atts
        ) = zip(*list(map(self._make_record, rows)))

        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        # The main inputs for bert
        masked_concepts = np.stack(pd.Series(masked_concepts) \
            .apply(convert_to_list_of_lists) \
            .apply(self._concept_tokenizer.encode) \
            .apply(lambda tokens: self._pad(tokens, padded_token=unused_token_id)))
        
        concepts = np.stack(pd.Series(concepts) \
            .apply(convert_to_list_of_lists) \
            .apply(self._concept_tokenizer.encode) \
            .apply(lambda tokens: self._pad(tokens, padded_token=unused_token_id)))
        
        pat_mask = (masked_concepts == unused_token_id).astype(int)

        time_interval_atts = np.asarray(
            self._concept_tokenizer.encode(
                np.stack(time_interval_atts).tolist()
            )
        )
        
        visit_masks = np.stack(visit_masks)
        
        visit_segments = np.stack(visit_segments)
        
        # The auxiliary inputs for bert
        dates = np.stack(
            pd.Series(dates) \
                .apply(convert_to_list_of_lists) \
                .apply(lambda time_stamps: self._pad(time_stamps, padded_token=0))
        )

        ages = np.stack(
            pd.Series(ages) \
                .apply(convert_to_list_of_lists) \
                .apply(lambda time_stamps: self._pad(time_stamps, padded_token=0))
        )

        input_dict = {'pat_seq': masked_concepts,
                      'pat_mask': pat_mask,
                      'pat_seq_time': dates,
                      'pat_seq_age': ages,
                      'visit_segment': visit_segments,
                      'visit_time_delta_att': time_interval_atts,
                      'visit_mask': visit_masks}
        
        concepts = np.reshape(concepts, (-1, self._max_num_of_concepts * self._max_num_of_visits))
        output_concept_masks = np.reshape(output_concept_masks, (-1, self._max_num_of_concepts * self._max_num_of_visits))
        output_visit_masks = np.ones_like(visit_concept_ids)
        output_dict = {'concept_predictions': np.stack([concepts, output_concept_masks],axis=-1),
                       'visit_predictions': np.stack([visit_concept_ids, output_visit_masks],
                                                     axis=-1)}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, _ = row_slicer

        concepts = row.concept_ids[start_index:end_index]
        dates = row.dates[start_index:end_index]
        ages = row.ages[start_index:end_index]
        visit_segments = row.visit_segments[start_index:end_index]
        visit_dates = row.visit_dates[start_index:end_index]
        visit_masks = row.visit_masks[start_index:end_index]
        visit_concept_ids = row.visit_concept_ids[start_index:end_index]
        # Skip the first element because there is no time interval for it
        time_interval_atts = row.time_interval_atts[start_index + 1:end_index]

        masked_concepts, output_concept_masks = zip(
            *list(map(self._mask_concepts, concepts)))

        return (
            output_concept_masks, masked_concepts, concepts, dates, ages,
            visit_concept_ids, visit_segments, visit_dates, visit_masks, time_interval_atts
        )

    def _mask_concepts(self, concepts):
        """
        Mask out 15% of the concepts
        :param concepts:
        :return:
        """
        masked_concepts = np.asarray(concepts).copy()
        output_mask = np.zeros((self._max_num_of_concepts,), dtype=int)

        if self._is_training:
            # the first position is reserved for cls, so we don't mask the first element           
            for word_pos in range(1, len(concepts)):
                if concepts[word_pos] == self._concept_tokenizer.get_unused_token_id():
                    break
                
                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_concepts[word_pos] = self._concept_tokenizer.get_mask_token_id()
                    elif dice < 0.9:
                        masked_concepts[word_pos] = random.randint(
                            self._concept_tokenizer.get_first_token_index(),
                            self._concept_tokenizer.get_last_token_index())
                    # else: 10% of the time we just leave the word as is
                    output_mask[word_pos] = 1

        return masked_concepts, output_mask


# %%
tokenizer = tokenize_concepts(patient_sequence,
                              'patient_concept_ids',
                              None,
                              tokenizer_path, 
                              encode=False)

# %%
tokenizer.fit_on_concept_sequences(patient_sequence.time_interval_atts)

# %%
visit_tokenizer = tokenize_concepts(patient_sequence,
                              'visit_concept_ids',
                              'visit_token_ids',
                              visit_tokenizer_path)

# %%

# %%
bert_data_generator = HierarchicalBertDataGenerator(training_data=patient_sequence, 
                                                    concept_tokenizer=tokenizer, 
                                                    visit_tokenizer=visit_tokenizer, 
                                                    max_num_of_visits=10, 
                                                    max_num_of_concepts=50, 
                                                    sliding_window=10, 
                                                    batch_size=8, 
                                                    max_seq_len=20*50, 
                                                    min_num_of_concepts=10)

# %%
batch_generator = bert_data_generator.create_batch_generator()

# %%
steps_per_epoch = bert_data_generator.get_steps_per_epoch()
dataset = tf.data.Dataset.from_generator(bert_data_generator.create_batch_generator,
                                     output_types=(bert_data_generator.get_tf_dataset_schema())) \
    .prefetch(tf.data.experimental.AUTOTUNE)

# %%
history = model.fit(dataset, 
                    steps_per_epoch=steps_per_epoch,
                    epochs=1)

# %%
