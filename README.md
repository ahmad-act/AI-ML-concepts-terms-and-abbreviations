# 📚 AI & ML Concepts, Terms, and Abbreviations

This glossary provides a comprehensive list of terms, abbreviations, and tools used in Artificial Intelligence (AI), Machine Learning (ML), and Large Language Models (LLMs).

---

## 🤖 Core AI & Machine Learning Terms

| Abbreviation | Term                         | Description                                                                 |
|--------------|------------------------------|-----------------------------------------------------------------------------|
| AI           | Artificial Intelligence       | Creating systems that simulate human intelligence.                          |
| ML           | Machine Learning              | Algorithms that learn from data without explicit programming.               |
| DL           | Deep Learning                 | ML using deep neural networks.                                              |
| NN           | Neural Network                | Inspired by the brain, used for pattern recognition in ML/DL.               |
| DNN          | Deep Neural Network           | NN with multiple hidden layers.                                             |
| CNN          | Convolutional Neural Network  | Specialized DNN for image and grid data.                                    |
| RNN          | Recurrent Neural Network      | NN for sequential data like time series and text.                           |
| LSTM         | Long Short-Term Memory        | RNN variant that captures long-term dependencies.                           |
| GRU          | Gated Recurrent Unit          | Lightweight LSTM alternative for sequences.                                 |
| GAN          | Generative Adversarial Network| Competing models generate realistic data.                                   |
| VAE          | Variational Autoencoder       | Learns latent representations for generative tasks.                         |
| RL           | Reinforcement Learning        | Learning via rewards from environment interaction.                          |
| DRL          | Deep Reinforcement Learning   | Combines RL and deep learning.                                              |

---

## 🧠 Large Language Models (LLMs) & NLP

| Abbreviation | Term                           | Description                                                                 |
|--------------|--------------------------------|-----------------------------------------------------------------------------|
| LLM          | Large Language Model           | Trained on vast text to understand/generate language.                      |
| NLP          | Natural Language Processing    | Enables computers to understand human language.                            |
| NLU          | Natural Language Understanding | Focuses on extracting meaning from text.                                   |
| NLG          | Natural Language Generation    | Generates coherent and relevant text.                                      |
| BERT         | Bidirectional Encoder Representations from Transformers | Contextual embeddings from both directions.    |
| RoBERTa      | Robustly Optimized BERT Approach| Improved BERT with better training.                                       |
| GPT          | Generative Pre-trained Transformer | Transformer model trained for language generation.                     |
| T5           | Text-to-Text Transfer Transformer | Reframes tasks as text-to-text.                                       |
| XLNet        | —                              | Combines benefits of BERT and autoregressive models.                      |
| Transformer  | —                              | NN using self-attention, basis for modern LLMs.                           |
| Attention    | —                              | Mechanism to focus on important parts of input.                           |
| Tokenizer    | —                              | Splits text into smaller processing units.                                |
| ST           | Sentence Transformer           | Maps sentences into dense embeddings.                                     |

---

## ⚙️ Model Formats & Quantization

| Abbreviation | Term                  | Description                                                                 |
|--------------|-----------------------|-----------------------------------------------------------------------------|
| GGUF         | GGUF Format           | Compact format for quantized models.                                        |
| GGML         | GPT-GGML Library      | C/C++ backend for efficient inference.                                      |
| FP16         | 16-bit Floating Point | Half-precision to reduce memory and speed up compute.                       |
| INT8         | 8-bit Integer         | Used in quantization to reduce model size.                                 |
| Q4_K_M       | Quantized 4-bit (KMeans)| Compact format for model inference.                                       |
| ONNX         | Open Neural Network Exchange | Cross-platform model format.                                        |
| TensorRT     | —                     | NVIDIA's high-performance inference engine.                                |
| TFLite       | TensorFlow Lite       | Lightweight inference framework for mobile/edge.                           |
| TF           | TensorFlow            | Google’s deep learning framework.                                          |
| PT           | PyTorch               | Meta’s (Facebook) widely used ML library.                                  |
| HF           | Hugging Face          | Platform and tools for ML model sharing and deployment.                    |

---

## 🛠️ Training & Development

| Term            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| Pre-train       | Initial training on broad data.                                             |
| Fine-tune       | Additional training on task-specific data.                                 |
| Epoch           | One pass through the entire dataset.                                       |
| Batch / Batch Size| Subset of data for a training iteration.                                |
| Loss            | Measures error between prediction and truth.                               |
| Optimizer       | Updates weights to minimize loss (e.g., SGD, Adam).                        |
| SGD             | Stochastic Gradient Descent – optimization method.                         |
| Adam            | Optimizer combining momentum and adaptive LR.                              |
| Learning Rate   | Size of weight updates during training.                                    |
| Backpropagation | Algorithm to update weights via gradients.                                 |
| Overfitting     | Model memorizes training data, poor generalization.                        |
| Underfitting    | Model fails to capture patterns in data.                                   |
| Regularization  | Prevents overfitting by adding constraints.                                |
| Hyperparameter  | Configurations like learning rate or batch size.                           |
| Gradient Clipping| Caps gradients to prevent explosion.                                     |
| Feature Engineering| Designing inputs to improve performance.                              |
| Transfer Learning| Using pre-trained models on new tasks.                                   |
| Embedding       | Vector representation of words, sentences, or images.                      |

---

## 📦 Tools & Libraries

| Term         | Description                                                               |
|--------------|---------------------------------------------------------------------------|
| Transformers | Hugging Face library of pre-trained models.                              |
| LangChain    | Toolkit for building LLM-based applications.                              |
| llama.cpp    | C/C++ LLM inference engine.                                               |
| Ollama       | Mac-native LLM runner.                                                    |
| vLLM         | High-throughput transformer inference.                                    |
| DeepSpeed    | Optimized training/inference library from Microsoft.                      |
| HF Hub       | Hugging Face’s model and dataset hub.                                     |
| TensorFlow   | Google's ML framework.                                                    |
| PyTorch      | Meta’s dynamic ML framework.                                              |
| Keras        | High-level NN API built on TensorFlow.                                    |
| FastAPI      | API backend for ML apps.                                                  |
| Streamlit    | Web app builder for ML in Python.                                         |
| Scikit-learn | Classical ML algorithms and tools.                                        |
| NumPy        | Numerical computation in Python.                                          |
| SciPy        | Scientific computing in Python.                                           |

---

## 🌐 Deployment & Serving

| Abbreviation | Term                     | Description                                                               |
|--------------|--------------------------|---------------------------------------------------------------------------|
| API          | Application Programming Interface | Interface to connect apps/services.                             |
| REST API     | Representational State Transfer | Standard web API architecture.                               |
| gRPC         | gRPC Remote Procedure Call | Efficient communication protocol.                               |
| GPU          | Graphics Processing Unit  | Hardware for fast parallel computation.                                |
| TPU          | Tensor Processing Unit    | Google’s AI-specific accelerator.                                       |
| Docker       | —                         | Container platform for deploying applications.                          |
| K8s          | Kubernetes                | Orchestrates containers (e.g., Docker).                                 |
| MLOps        | Machine Learning Operations| Tools/practices for ML lifecycle in production.                         |
| Inference    | —                         | Using a trained model to make predictions.                              |
| Deployment   | —                         | Making a model available in production.                                 |

---

## 🧪 Evaluation Metrics

| Abbreviation | Term                          | Description                                                                |
|--------------|-------------------------------|----------------------------------------------------------------------------|
| Accuracy     | —                              | Proportion of correct predictions.                                        |
| Precision    | —                              | True positives / predicted positives.                                     |
| Recall       | —                              | True positives / actual positives.                                        |
| F1 Score     | —                              | Harmonic mean of precision and recall.                                    |
| BLEU         | Bilingual Evaluation Understudy| Compares generated vs. reference translations.                            |
| ROUGE        | Recall-Oriented Understudy for Gisting Evaluation | Overlap in summarization tasks.         |
| METEOR       | Metric for Evaluation of Translation with Explicit ORdering | Translation evaluation metric.      |
| Perplexity   | —                              | How well a model predicts text (lower = better).                          |

---

## 🧩 AI Organizations & Contributors

| Organization             | Focus                                                                |
|--------------------------|----------------------------------------------------------------------|
| OpenAI                   | ChatGPT, AGI research.                                                |
| Anthropic                | Claude models, AI alignment.                                         |
| Google DeepMind          | RL, robotics, AGI.                                                    |
| Meta AI (FAIR)           | LLaMA models, open-source AI.                                        |
| Microsoft Research       | Azure AI, responsible AI, OpenAI partnership.                        |
| HuggingFace              | Transformers, model sharing.                                         |
| EleutherAI               | Open-source LLMs (GPT-J, GPT-Neo).                                   |
| Stability AI             | Generative AI (e.g., Stable Diffusion).                             |
| Together AI              | Open model hosting and training.                                     |
| Llama.cpp                | Lightweight CPU-based inference.                                     |
| Ollama                   | Local LLMs on Apple Silicon.                                         |
| Mistral AI               | High-performance open models (e.g., Mistral 7B).                     |
| xAI                      | Elon Musk’s AI company focused on AGI.                              |
| Baidu ERNIE Bot          | Chinese language LLMs.                                               |
| Alibaba Cloud (Qwen)     | Chinese LLMs for general use.                                        |
| Tencent HunYuan          | Multilingual LLM and general AI.                                     |
