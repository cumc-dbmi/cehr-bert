#!/usr/bin/env python
# coding: utf-8
# %%
import tensorflow as tf


# %%
from models.custom_layers import *
import numpy as np


# %%
num_concept_per_v = 50
num_visit = 20
num_seq = num_concept_per_v*num_visit


concept_vocab_size = 40000
visit_vocab_size = 10

embeddinig_size = 128
time_embeddings_size = 16
depth = 16
num_heads = 8
transformer_dropout: float = 0.1
embedding_dropout: float = 0.6
l2_regularizer = tf.keras.regularizers.l2(1e-4)
    
identity = tf.constant(np.insert(np.identity(num_visit), range(1, num_visit), 0, axis=1), dtype=tf.float32)
identity_inverse = tf.constant(np.insert(np.identity(num_visit-1), range(0, num_visit), 0, axis=1), dtype=tf.float32)


# %%
concepts = tf.random.uniform((1, 1000), dtype=tf.int32, minval=1, maxval=1000)
time_stamps = tf.sort(tf.random.uniform((1, 1000), dtype=tf.int32, maxval=1000))
ages = tf.sort(tf.random.uniform((1, 1000), dtype=tf.int32, minval=18, maxval=80))
mask = tf.sort(tf.random.uniform((1, 1000), dtype=tf.int32, maxval=2))

visit_time_stamps = tf.sort(tf.random.uniform((1, 20), dtype=tf.int32, maxval=1000))
visit_seq_time_delta = tf.sort(tf.random.uniform((1, 19), dtype=tf.int32, maxval=1000))
visit_mask = tf.sort(tf.random.uniform((1, 20), dtype=tf.int32, maxval=2))

# %%
pt_seq = tf.keras.layers.Input(shape=(num_seq,), dtype='int32',
name='patient_seq')
pt_seq_age = tf.keras.layers.Input(shape=(num_seq,), dtype='int32',
name='patient_seq_age')
pt_seq_time = tf.keras.layers.Input(shape=(num_seq,), dtype='int32',
name='patient_seq_time')
mask = tf.keras.layers.Input(shape=(num_seq,), dtype='int32', name='mask')

visit_seq_age = tf.keras.layers.Input(shape=(num_visit,), dtype='int32',
name='visit_seq_age')
visit_seq_time = tf.keras.layers.Input(shape=(num_visit,), dtype='int32',
name='visit_seq_time')
visit_seq_time_delta = tf.keras.layers.Input(shape=(num_visit - 1,), dtype='int32',
name='visit_seq_time_delta')

visit_mask = tf.keras.layers.Input(shape=(num_visit,), dtype='int32', 
                                   name='visit_mask')

default_inputs = [pt_seq, pt_seq_age, pt_seq_time, mask, visit_seq_time_delta, visit_mask]

pt_seq = tf.reshape(pt_seq, (-1, num_visit, num_concept_per_v))
pt_seq_age = tf.reshape(pt_seq_age, (-1, num_visit, num_concept_per_v))
pt_seq_time = tf.reshape(pt_seq_time, (-1, num_visit, num_concept_per_v))


# %%
#concept_embedding_layer = tf.keras.layers.Embedding(vocab_size, embeddinig_size)

#l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)

#output the embedding_matrix:
concept_embedding_layer = ReusableEmbedding(
    concept_vocab_size, embeddinig_size,
    input_length=num_seq,
    name='bpe_embeddings'
    # Regularization is based on paper "A Comparative Study on
    # Regularization Strategies for Embedding-based Neural Networks"
    # https://arxiv.org/pdf/1508.03721.pdf
    #embeddings_regularizer=l2_regularizer
)
 # # define the time embedding layer for absolute time stamps (since 1970)
time_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                                  name='time_embedding_layer')
# define the age embedding layer for the age w.r.t the medical record
age_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                                 name='age_embedding_layer')

temporal_transformation_layer = tf.keras.layers.Dense(embeddinig_size,
                                                            activation='tanh',
                                                            name='scale_pat_seq_layer')


# %%
pt_seq_concept_embeddings, embedding_matrix = concept_embedding_layer(pt_seq)
pt_seq_age_embeddings = time_embedding_layer(pt_seq_age)
pt_seq_time_embeddings = time_embedding_layer(pt_seq_time)

# dense layer for rescale the patient sequence embeddings back to the original size
input_for_encoder = temporal_transformation_layer(
            tf.concat([pt_seq_concept_embeddings, pt_seq_age_embeddings, pt_seq_time_embeddings],
                      axis=-1, name='concat_for_encoder'))

# %%
batch_size, num_visits, num_concepts, _ = tf.shape(pt_seq_concept_embeddings)
pt_seq_concept_embeddings = tf.reshape(tf.transpose(pt_seq_concept_embeddings, perm=[0, 2, 1, 3]), (-1, num_concepts, embeddinig_size))
concept_mask = create_concept_mask(tf.reshape(mask, (-1, num_concepts)), num_concepts)

merged_visit_mask = tf.reshape(tf.stack([visit_mask, visit_mask], axis=2), (-1, num_visits * 2))[:, 1:]
expanded_visit_mask = create_concept_mask(merged_visit_mask, num_visits * 2 - 1)

# %%
encoder = Encoder(name='encoder',
                  num_layers=depth,
                  d_model=embeddinig_size,
                  num_heads=num_heads,
                  dropout_rate=transformer_dropout)


# %%
contextualized_concept_emebddings, attention_weights = encoder(x_reshape, concept_mask)

# %%
contextualized_visit_emebddings = tf.reshape(contextualized_concept_emebddings, (-1, 20, 50, embeddinig_size))

# %%
visit_embeddings = contextualized_visit_emebddings[:, :, 0]

# %%
contextualized_concept_emebddings_reshaped = tf.reshape(contextualized_concept_emebddings, 
                                                        (-1, num_seq, embeddinig_size))

# %%
att_embeddings, _ = concept_embedding_layer(visit_seq_time_delta)

# %%
expanded_visit_embeddings = tf.transpose(tf.transpose(visit_embeddings, perm=[0, 2, 1]) @ identity, perm=[0, 2, 1])
expanded_att_embeddings = tf.transpose(tf.transpose(att_embeddings, perm=[0, 2, 1]) @ identity_inverse, perm=[0, 2, 1])
merged_visit_embeddings = expanded_visit_embeddings + expanded_att_embeddings

# %%
encoder_2 = Encoder(name='visit_encoder',
                  num_layers=depth,
                  d_model=embeddinig_size,
                  num_heads=num_heads,
                  dropout_rate=transformer_dropout)

# %%
contextualized_visit_embeddings, _ = encoder_2(merged_visit_embeddings, expanded_visit_mask)

# %%
mha = MultiHeadAttention(embeddinig_size, num_heads)

# %%
concept_embeddings,_ = mha(contextualized_visit_embeddings, 
                         contextualized_visit_embeddings, 
                         contextualized_concept_emebddings_reshaped, 
                         merged_visit_mask, None)

# %%
output_layer = TiedOutputEmbedding(
    projection_regularizer=l2_regularizer,
    projection_dropout=embedding_dropout,
    name='concept_prediction_logits')

visit_prediction_dense = tf.keras.layers.Dense(visit_vocab_size)

concept_softmax_layer = tf.keras.layers.Softmax(name='concept_predictions')
visit_softmax_layer = tf.keras.layers.Softmax(name='visit_predictions')

# %%
concept_predictions = concept_softmax_layer(
    output_layer([concept_embeddings, embedding_matrix]))

visit_predictions = visit_softmax_layer(visit_prediction_dense(contextualized_visit_embeddings))

# %%
model = tf.keras.Model(
    inputs=default_inputs,
    outputs=[concept_predictions, visit_predictions])

# %%
model.summary()

# %%
#visit_seq_time

# %% [markdown]
# ### Calculate time delta and insert into visit sequence

# %%
visit_seq_time_delta = tf.concat([visit_seq_time[:, 1:] - visit_seq_time[:, :-1]], axis=-1)

# %%
visit_seq_time_delta

# %%
att_embeddings, _ = concept_embedding_layer(visit_seq_time_delta)

# %%
A = np.identity(num_visit)
identity = tf.constant(np.insert(A, range(1, num_visit), 0, axis=1), dtype=tf.float32)
expanded_visit_embeddings = tf.transpose(tf.transpose(visit_embeddings, perm=[0, 2, 1]) @ identity, perm=[0, 2, 1])

B = np.identity(num_visit-1)
identity_inverse = tf.constant(np.insert(B, range(0, num_visit), 0, axis=1), dtype=tf.float32)

expanded_att_embeddings = tf.transpose(tf.transpose(att_embeddings, perm=[0, 2, 1]) @ identity_inverse, perm=[0, 2, 1])
merged_visit_embeddings = expanded_visit_embeddings + expanded_att_embeddings

# %%
merged_visit_embeddings
