import flwr as fl
import tensorflow as tf

# ===============================
# CONFIG
# ===============================
TRAIN_DIR = "Dataset_split/train"
VAL_DIR = "Dataset_split/val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 2
ROUNDS = 5

class_weights = {0: 5.0, 1: 0.4, 2: 2.0}

# ===============================
# DATA LOADING
# ===============================
from tensorflow.keras.applications.densenet import preprocess_input


def load_client_data():
    ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",  
    )

    ds = ds.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.cache().prefetch(tf.data.AUTOTUNE)


def load_val_data():
    ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical", 
    )

    ds = ds.map(
        lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    return ds.cache().prefetch(tf.data.AUTOTUNE)


# ===============================
# MODEL
# ===============================
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)

        return tf.reduce_sum(weight * cross_entropy, axis=1)

    return loss


def create_model():
    base = tf.keras.applications.DenseNet121(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3)
    )

    # Fine-tuning
    for layer in base.layers[:-30]:
        layer.trainable = False

    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=focal_loss(),
        metrics=["accuracy"],
        run_eagerly=False,
    )

    return model


# ===============================
# FLOWER CLIENT
# ===============================
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = create_model()
        self.train_data = load_client_data()
        self.val_data = load_val_data()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        self.model.fit(
            self.train_data, epochs=EPOCHS, class_weight=class_weights, verbose=1
        )

        # ✅ Correct sample count (important for FedAvg)
        num_examples = sum(1 for _ in self.train_data) * BATCH_SIZE

        # 🔥 Save only final global model
        if config["server_round"] == ROUNDS:
            self.model.save("final_model_client.h5")
            print("✅ Final Global Model Saved!")

        return self.model.get_weights(), num_examples, {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss, acc = self.model.evaluate(self.val_data, verbose=0)

        num_examples = sum(1 for _ in self.val_data) * BATCH_SIZE

        return loss, num_examples, {"accuracy": acc}


# ===============================
# START CLIENT
# ===============================
fl.client.start_numpy_client(
    server_address="172.16.104.164:8080", client=FlowerClient()
)
