

# 🎵 Music Generator – README

This project is a **character-level music generation model** built using **TensorFlow** and inspired by the *MIT Introduction to Deep Learning* lab.
It trains an LSTM-based neural network on a dataset of musical sequences and generates new music in the same style.

---

## 📌 Features

* Loads musical text sequences and prepares a character-level vocabulary
* Vectorizes input sequences for training
* Builds and trains an **LSTM-based deep learning model**
* Generates new musical sequences character-by-character
* Integrates with **Comet ML** for experiment tracking (optional)
* Supports running in **Google Colab** with Drive integration

---

## 📁 Project Structure

```
music_generator.ipynb
├── Google Drive mounting (optional)
├── Comet ML setup
├── Data loading + cleaning
├── Character vectorization
├── LSTM model building
├── Training loop
└── Music generation function
```

---

## 🛠️ Requirements

Install required packages:

```bash
pip install tensorflow comet_ml
```

Optional (if using Colab):

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 🚀 How to Use

### **1. Set up Comet ML (Optional)**

Inside the notebook, enter your Comet API key:

```python
experiment = Experiment(
    api_key="YOUR_API_KEY",
    project_name="music-generator",
    workspace="your_workspace",
)
```

If you don’t want Comet tracking, you can comment out the experiment code.

---

### **2. Load and Prepare Text Data**

The notebook joins your songs into a single text corpus:

```python
joined_songs = "\n".join(songs)
vocab = sorted(set(joined_songs))
```

---

### **3. Vectorize Characters**

The model creates mapping dictionaries:

```python
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
```

---

### **4. Build & Train the Model**

A multi-layer LSTM neural network is created:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 256),
    tf.keras.layers.LSTM(1024, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])
```

Training is performed using teacher forcing.

---

### **5. Generate New Music**

After training:

```python
mdl.lab1.play_song(example_song)
```

Or generate new characters using the custom generation loop.

---

## 🎶 Output

The model outputs **new music text sequences**, which can optionally be played back using helper functions from the MIT deep learning labs.

---

## 📈 Experiment Tracking (Optional)

Comet ML tracks:

* Training/validation loss
* Model metrics
* Hyperparameters
* Code snapshots

---
