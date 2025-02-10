EVAL_SPLIT = 0.01  # Fraction of training data used for evaluation
WANDB_KEY = "<YOUR_WANDB_KEY>"  # Weights and Biases API key

# -------------------- Configuration for M3 Training --------------------
M3_TRAIN_FOLDERS = [
    "<YOUR_TRAINING_DATA_FOLDER>"  # Directory containing training data for M3
]

M3_EVAL_FOLDERS = [
    "<YOUR_EVALUATION_DATA_FOLDER>"  # Directory containing evaluation data for M3 (optional)
]

PATCH_SIZE = 64  # Size of each patch
PATCH_LENGTH = 512  # Length of the patches
PATCH_NUM_LAYERS = 12  # Number of layers in the encoder
TOKEN_NUM_LAYERS = 3  # Number of layers in the decoder
M3_HIDDEN_SIZE = 768  # Size of the hidden layer

M3_NUM_EPOCH = 100  # Maximum number of epochs for training
M3_LEARNING_RATE = 1e-4  # Learning rate for the optimizer
M3_BATCH_SIZE = 16  # Batch size per GPU (single card) during training
M3_MASK_RATIO = 0.45  # Ratio of masked elements during training
M3_DETERMINISTIC = True  # Ensures deterministic results with random seeds
M3_WANDB_LOG = True  # Enable logging to Weights and Biases
M3_LOAD_CKPT = True  # Load model weights from a checkpoint if available

M3_WEIGHTS_PATH = (
    "weights_m3"+
    "_h_size_" + str(M3_HIDDEN_SIZE) +
    "_t_layers_" + str(TOKEN_NUM_LAYERS) +
    "_p_layers_" + str(PATCH_NUM_LAYERS) +
    "_p_size_" + str(PATCH_SIZE) +
    "_p_length_" + str(PATCH_LENGTH) +
    "_lr_" + str(M3_LEARNING_RATE) +
    "_batch_" + str(M3_BATCH_SIZE) +
    "_mask_" + str(M3_MASK_RATIO) + ".pth"
)  # Path to store the model weights
M3_LOGS_PATH = M3_WEIGHTS_PATH.replace("weights", "logs").replace("pth", "txt")  # Path to save training logs

# -------------------- Configuration for CLaMP3 Training ----------------
CLAMP3_TRAIN_JSONL = "<YOUR_TRAINING_JSONL_FILE>"  # Path to the JSONL file with training data for CLaMP3
CLAMP3_EVAL_JSONL = "<YOUR_EVALUATION_JSONL_FILE>"  # Path to the JSONL file with evaluation data for CLaMP3 (optional)

CLAMP3_HIDDEN_SIZE = 768  # Size of the hidden layer
TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"  # Name of the pre-trained text model
MAX_TEXT_LENGTH = 128  # Maximum allowed length for text input

AUDIO_HIDDEN_SIZE = 768  # Size of the hidden layer for audio features
AUDIO_NUM_LAYERS = 12  # Number of layers in the audio encoder
MAX_AUDIO_LENGTH = 128  # Maximum allowed length for audio input

CLAMP3_NUM_EPOCH = 100  # Maximum number of epochs for training
CLAMP3_LEARNING_RATE = 1e-5  # Learning rate for the optimizer
CLAMP3_BATCH_SIZE = 256  # Batch size per GPU (single card) during training
LOGIT_SCALE = 1  # Scaling factor for contrastive loss

FREEZE_TEXT = False  # Freeze the weights of the text model and text projection layer
TEXT_DROPOUT = True  # Whether to apply dropout during text processing
CLAMP3_DETERMINISTIC = True  # Ensures deterministic results with random seeds
CLAMP3_LOAD_M3 = True  # Load weights from the M3 model
CLAMP3_WANDB_LOG = True  # Enable logging to Weights and Biases
CLAMP3_LOAD_CKPT = True  # Load weights from a checkpoint if available
SAVE_EVERY = 5  # Save model weights every SAVE_EVERY epochs

CLAMP3_WEIGHTS_PATH = (
    "weights_clamp3_saas" +
    "_h_size_" + str(CLAMP3_HIDDEN_SIZE) +
    "_t_model_" + TEXT_MODEL_NAME.replace("/", "_") +
    "_t_length_" + str(MAX_TEXT_LENGTH) +
    "_a_size_" + str(AUDIO_HIDDEN_SIZE) +
    "_a_layers_" + str(AUDIO_NUM_LAYERS) +
    "_a_length_" + str(MAX_AUDIO_LENGTH) +
    "_s_size_" + str(M3_HIDDEN_SIZE) +
    "_s_layers_" + str(PATCH_NUM_LAYERS) +
    "_p_size_" + str(PATCH_SIZE) +
    "_p_length_" + str(PATCH_LENGTH) + ".pth"

)  # Path to store CLaMP3 model weights
CLAMP3_LOGS_PATH = CLAMP3_WEIGHTS_PATH.replace("weights", "logs").replace("pth", "txt")  # Path to save training logs
