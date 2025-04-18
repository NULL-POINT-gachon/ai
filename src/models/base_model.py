from tensorflow.keras import layers, Model

def build_model(vocab_sizes):
    """
    vocab_sizes: dict 형태로 '컬럼명': vocab_size
    """
    inputs = {}
    embeddings = []

    def embed_input(name, vocab_size, emb_dim):
        inp = layers.Input(shape=(1,), name=name)
        emb = layers.Embedding(vocab_size, emb_dim)(inp)
        emb = layers.Reshape((emb_dim,))(emb)
        inputs[name] = inp
        embeddings.append(emb)

    # 임베딩 입력
    embed_input('user_id', vocab_sizes['user_id'], 32)
    embed_input('item_id', vocab_sizes['item_id'], 32)
    embed_input('TRAVEL_MOTIVES', vocab_sizes['TRAVEL_MOTIVES'], 8)
    embed_input('TRAVEL_STYLES', vocab_sizes['TRAVEL_STYLES'], 8)
    embed_input('TRAVEL_PURPOSE', vocab_sizes['TRAVEL_PURPOSE'], 8)

    # 숫자형 입력
    for col in ['GENDER', 'AGE_GRP', 'TRAVEL_COMPANIONS_NUM', 'VISIT_CHC_REASON_CD']:
        inp = layers.Input(shape=(1,), name=col)
        inputs[col] = inp
        embeddings.append(inp)

    x = layers.Concatenate()(embeddings)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=list(inputs.values()), outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
