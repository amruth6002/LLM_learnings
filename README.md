# LLM from Scratch

This repository contains a Jupyter Notebook, `llm-from-scratch.ipynb`, detailing the implementation of a GPT-like Large Language Model (LLM) from the ground up using PyTorch. The code is a practical, hands-on guide to understanding the core components of modern transformer-based language models.

This implementation is inspired by and based on the concepts from Sebastian Raschka's "[Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)" book.

##  Key Features

- **Modular Components**: Each part of the GPT architecture is built as a separate, understandable class.
- **Self-Attention Mechanism**: A detailed implementation of Multi-Head Self-Attention, the core of the Transformer model.
- **Transformer Blocks**: Combines multi-head attention and feed-forward neural networks into a complete Transformer block.
- **Complete GPT Model**: Assembles the components into a full GPT model, including token and positional embeddings.
- **Configuration-Driven**: Model hyperparameters are managed through a central configuration dictionary, making it easy to experiment with different model sizes.
- **Tokenization**: Utilizes `tiktoken`, OpenAI's fast BPE tokenizer, for text processing.

##  Model Architecture

The model implemented in this notebook is based on the GPT-2 architecture. The key components are:

1.  **Embedding Layer**:
    -   **Token Embeddings**: Converts input tokens into dense vectors.
    -   **Positional Embeddings**: Adds positional information to the token embeddings, allowing the model to understand word order.

2.  **Transformer Blocks**:
    -   The model consists of multiple stacked Transformer blocks. Each block contains:
        -   **Multi-Head Attention**: An improved version of the self-attention mechanism that allows the model to focus on different parts of the input sequence simultaneously. It includes a causal mask to ensure that predictions for a token can only depend on previous tokens.
        -   **Layer Normalization**: Applied before the main sub-layers to stabilize training.
        -   **Feed-Forward Network**: A two-layer fully connected network with a GELU activation function.
        -   **Residual Connections**: "Shortcut" connections that add the input of a sub-layer to its output, helping to prevent the vanishing gradient problem.

3.  **Final Output Layer**:
    -   A final layer normalization is applied, followed by a linear layer that maps the transformer's output to the vocabulary size, producing the final logits.

*(You can create and insert a diagram of the overall model architecture here.)*
`[Image of the complete GPT Model Architecture]`

### Multi-Head Attention

The attention mechanism calculates a weighted score for each token in the input sequence relative to every other token. The "multi-head" variant performs this process in parallel multiple times with different linear projections, capturing various types of relationships in the data.

*(A diagram illustrating the query, key, and value computations would be beneficial here.)*
`[Image of the Multi-Head Attention mechanism]`

## ⚙️ Model Configuration

The notebook uses a configuration dictionary, `GPT_CONFIG_124M`, which mirrors the hyperparameters of the original 124 million parameter GPT-2 model.

```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}
```

The model built with this configuration has approximately **163 million parameters** and would have a size of about **622 MB**.

##  How to Use

### Prerequisites

You need to have Python, PyTorch, and `tiktoken` installed.

```bash
pip install torch tiktoken jupyterlab
```

### Running the Notebook

1.  Clone the repository:
    ```bash
    git clone https://github.com/amruth6002/LLM_from_Scratch.git
    cd LLM_from_Scratch
    ```

2.  Start Jupyter Lab:
    ```bash
    jupyter lab
    ```

3.  Open `llm-from-scratch.ipynb` and run the cells. The notebook is structured to be executed sequentially.

### Code Structure

The notebook is organized into the following sections:
1.  **Setup**: Imports necessary libraries.
2.  **Tokenization**: Initializes the `tiktoken` tokenizer.
3.  **Multi-Head Attention**: Implements the `MultiHeadAttention` class.
4.  **GPT Model Architecture**: Defines the core classes for the model:
    - `LayerNorm`
    - `GELU`
    - `FeedForward`
    - `TransformerBlock`
    - `GPTModel` (The main model class)
5.  **Model Initialization**: Instantiates the model using the `GPT_CONFIG_124M` configuration and calculates its size.
6.  **Training and Evaluation (Commented Out)**: The notebook includes commented-out sections for a data loader, training loop, and evaluation functions. You can uncomment and adapt these to train the model on your own dataset.

I hope this README is helpful for your project! Let me know if you need any other assistance.

