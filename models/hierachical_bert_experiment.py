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
    # Calculate the patient sequence length
    max_seq = num_of_visits * num_of_concepts

    pat_seq = tf.keras.layers.Input(shape=(max_seq,), dtype='int32', name='pat_seq')
    pat_seq_age = tf.keras.layers.Input(shape=(max_seq,), dtype='int32', name='pat_seq_age')
    pat_seq_time = tf.keras.layers.Input(shape=(max_seq,), dtype='int32', name='pat_seq_time')
    pat_mask = tf.keras.layers.Input(shape=(max_seq,), dtype='int32', name='pat_mask')

    visit_time_delta_att = tf.keras.layers.Input(shape=(num_of_visits - 1,), dtype='int32',
                                                 name='visit_time_delta_att')
    visit_mask = tf.keras.layers.Input(shape=(num_of_visits,), dtype='int32', name='visit_mask')

    default_inputs = [pat_seq, pat_seq_age, pat_seq_time, pat_mask,
                      visit_time_delta_att, visit_mask]

    # Reshape the data into the visit view (batch, num_of_visits, num_of_concepts)
    pat_seq = tf.reshape(pat_seq, (-1, num_of_visits, num_of_concepts))
    pat_seq_age = tf.reshape(pat_seq_age, (-1, num_of_visits, num_of_concepts))
    pat_seq_time = tf.reshape(pat_seq_time, (-1, num_of_visits, num_of_concepts))

    pat_concept_mask = tf.reshape(pat_mask, (-1, num_of_concepts))[:, tf.newaxis, tf.newaxis, :]

    visit_mask_with_att = tf.reshape(tf.stack([visit_mask, visit_mask], axis=2),
                                     (-1, num_of_visits * 2))[:, 1:]

    visit_concept_mask = visit_mask_with_att[:, tf.newaxis, tf.newaxis, :]

    # output the embedding_matrix:
    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size, embedding_size,
        input_length=max_seq,
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
        shape=(-1, max_seq, embeddinig_size)
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
num_visit = 20
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
                                            concept_vocab_size,
                                            visit_vocab_size,
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

pat_seq_input = tf.random.uniform((n_of_data_points, num_seq), maxval=100, dtype=tf.int32)
pat_seq_age_input = tf.sort(tf.random.uniform((n_of_data_points, num_seq), minval=18, maxval=100, dtype=tf.int32))
pat_seq_time_input = tf.sort(tf.random.uniform((n_of_data_points, num_seq), maxval=1000, dtype=tf.int32))
pat_mask_input = tf.sort(tf.random.uniform((n_of_data_points, num_seq), maxval=2, dtype=tf.int32))

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
dataset = tf.data.Dataset.from_tensor_slices((inputs, output_dict)).cache().batch(1)

# %%
for x, y in dataset:
    print(model(x))
    break

# %%
model.fit(dataset)

# %%
