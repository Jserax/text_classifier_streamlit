import keras_nlp
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import bentoml

EPOCHS = 5
BATCH_SIZE = 64
SEQ_LENGTH = 200
VOCAB_SIZE = 20000
EMBEDDING_DIM = 32

ENCODER_NUMS = 2
NUM_HEADS = 4
INTER_DIM = 128
DROPOUT = 0.5
CLASS_WEIGHTS = {0: 0.61588777, 1: 2.65725961}


def train_model():
    """
    train model and save in to bentoml
    """
    train = pd.read_csv('data/processed/train.csv')
    test = pd.read_csv('data/processed/test.csv')
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train['comment'].to_numpy(),
         train['toxic'].to_numpy())).batch(BATCH_SIZE)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (test['comment'].to_numpy(),
         test['toxic'].to_numpy())).batch(BATCH_SIZE)

    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(
                train_ds.map(lambda x, y: x),
                vocabulary_size=VOCAB_SIZE,
                lowercase=True,
                strip_accents=True,
                reserved_tokens=["[PAD]", "[START]", "[END]", "[UNK]"],)

    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
                vocabulary=vocab,
                lowercase=True,
                strip_accents=True,
                oov_token="[UNK]",)

    packer = keras_nlp.layers.StartEndPacker(
                start_value=tokenizer.token_to_id("[START]"),
                end_value=tokenizer.token_to_id("[END]"),
                pad_value=tokenizer.token_to_id("[PAD]"),
                sequence_length=SEQ_LENGTH,)

    embedding = keras_nlp.layers.TokenAndPositionEmbedding(
                vocabulary_size=VOCAB_SIZE,
                sequence_length=SEQ_LENGTH,
                embedding_dim=EMBEDDING_DIM,)

    input = keras.Input(shape=(), dtype='string')
    x = tokenizer(input)
    x = packer(x)
    x = embedding(x)
    for _ in range(ENCODER_NUMS):
        x = keras_nlp.layers.TransformerEncoder(
                num_heads=NUM_HEADS,
                intermediate_dim=INTER_DIM,
                dropout=DROPOUT)(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=input, outputs=outputs)
    lr = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=3e-4,
                decay_steps=3284,
                decay_rate=0.5,
                staircase=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision()])
    model.fit(train_ds, epochs=EPOCHS,
              validation_data=test_ds,
              class_weight=CLASS_WEIGHTS)
    bentoml.keras.save_model("my_keras_model", model)


if __name__ == "__main__":
    train_model()
