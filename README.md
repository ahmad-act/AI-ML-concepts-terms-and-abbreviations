# 📚 AI & ML concepts, terms and abbreviations

This glossary provides a comprehensive list of key terms, abbreviations, and tools commonly used in Artificial Intelligence (AI), Machine Learning (ML), and Large Language Models (LLMs). It's a helpful reference for both beginners and experienced practitioners navigating the AI landscape.

---

## 🤖 Core AI & Machine Learning Terms

| Abbreviation | Term                         | Description                                                                 |
|--------------|------------------------------|-----------------------------------------------------------------------------|
| AI           | Artificial Intelligence       | The broad field of creating intelligent machines.                           |
| ML           | Machine Learning              | A subfield of AI focused on algorithms that learn from data.               |
| DL           | Deep Learning                 | A subset of ML using neural networks with multiple layers.                 |
| NLP          | Natural Language Processing   | AI dealing with human language understanding and generation.              |
| CV           | Computer Vision               | AI that interprets visual information from images/videos.                 |
| RL           | Reinforcement Learning        | ML where agents learn by trial and error with rewards/punishments.       |
| GAN          | Generative Adversarial Network| Competing neural nets generating realistic data.                          |
| RNN          | Recurrent Neural Network      | Neural net for sequential data like text or time series.                  |
| CNN          | Convolutional Neural Network  | Neural network used primarily in image recognition.                       |
| LSTM         | Long Short-Term Memory        | A type of RNN designed to remember long-term dependencies.                |
| GNN          | Graph Neural Network          | Processes graph-structured data.                                          |
| VAE          | Variational Autoencoder       | Type of generative model that learns latent representations.             |

---

## 🧠 Large Language Models (LLMs)

| Abbreviation | Term                           | Description                                                                 |
|--------------|--------------------------------|-----------------------------------------------------------------------------|
| LLM          | Large Language Model           | AI trained on vast text to generate and understand natural language.       |
| SFT          | Supervised Fine-Tuning         | Training a pre-trained LLM further with labeled data.                      |
| RLHF         | Reinforcement Learning from Human Feedback | Aligns LLM behavior using human feedback.                          |
| LoRA         | Low-Rank Adaptation            | Efficient fine-tuning with reduced trainable parameters.                   |
| MoE          | Mixture of Experts             | Model architecture with specialized expert modules.                        |
| PEFT         | Parameter-Efficient Fine-Tuning| Efficient fine-tuning method like LoRA.                                    |
| IFT          | Instruction Fine-Tuning        | Training LLMs to better follow instructions.                               |
| ST           | Sentence Transformer           | Models that encode text into embeddings for similarity tasks.              |
| LM           | Language Model                 | General category including LLMs and smaller models.                        |
| AGI          | Artificial General Intelligence| Hypothetical AI matching human intelligence in any task.                   |
| TTS          | Text-to-Speech                 | Converts text into spoken audio.                                           |
| STT          | Speech-to-Text                 | Converts spoken audio into text.                                           |
| OCR          | Optical Character Recognition  | Extracts text from images or scanned documents.                            |

---

## ⚙️ Model Formats & Quantization

| Abbreviation | Term                  | Description                                                                 |
|--------------|-----------------------|-----------------------------------------------------------------------------|
| GGUF         | GGUF Format           | Efficient binary format for LLM inference (used with GGML).                |
| GGML         | GPT-GGML Library      | C++/C-based framework for quantized LLMs.                                  |
| FP16         | 16-bit Floating Point | Reduced precision format for performance.                                  |
| INT8         | 8-bit Integer         | Lower precision weights for faster inference and less memory.             |
| Q4_K_M       | Quantized 4-bit (KMeans)| Specific 4-bit quantization format used in GGUF.                          |
| ONNX         | Open Neural Network Exchange | Interoperable format across ML frameworks.                          |
| TF           | TensorFlow            | Google’s deep learning framework.                                          |
| PT           | PyTorch               | Meta’s (Facebook) widely used ML library.                                  |
| HF           | Hugging Face          | Platform and tools for ML model sharing and deployment.                    |

---

## 🛠️ Training & Development

| Term            | Description                                                                 |
|-----------------|-----------------------------------------------------------------------------|
| Fine-tune       | Adapting a pre-trained model to a specific task or domain.                 |
| Pre-train       | Training a model on general data before specific task fine-tuning.         |
| Epoch           | One complete pass through the training dataset.                            |
| Batch Size      | Number of samples used in one training step.                               |
| Loss            | Indicator of how well the model's predictions match the true values.       |
| Optimizer       | Algorithm (e.g., SGD, Adam) that updates model weights during training.    |
| Learning Rate   | Controls the step size for weight updates.                                 |
| Overfitting     | When the model memorizes rather than generalizes.                          |
| Underfitting    | When the model fails to learn the patterns in the training data.           |
| Validation Set  | Data used to monitor performance during training.                          |
| Test Set        | Final data used to evaluate model after training.                          |

---

## 📦 Tools & Libraries

| Term         | Description                                                               |
|--------------|---------------------------------------------------------------------------|
| LangChain    | Framework for building LLM-powered applications.                          |
| Transformers | Hugging Face library of pre-trained models.                              |
| llama.cpp    | C/C++ implementation of LLMs for local inference.                        |
| Ollama       | Tool for running LLMs locally, optimized for Mac.                         |
| vLLM         | High-throughput LLM inference engine.                                    |
| DeepSpeed    | Microsoft library for optimizing training/inference of large models.      |
| HF Hub       | Centralized hub to publish and access ML models.                         |
| FastAPI      | High-performance API framework, often used with AI services.             |
| Streamlit    | Tool for creating interactive ML web apps with Python.                   |

---

## 🌐 Deployment & Serving

| Abbreviation | Term                     | Description                                                               |
|--------------|--------------------------|---------------------------------------------------------------------------|
| API          | Application Programming Interface | Way for programs to interact with ML services.                   |
| REST API     | Representational State Transfer | Standard architecture for web-based APIs.                      |
| gRPC         | gRPC Remote Procedure Call | High-performance method for client-server communication.         |
| GPU          | Graphics Processing Unit  | Hardware accelerator used in AI computation.                            |
| TPU          | Tensor Processing Unit    | Google’s custom hardware for ML workloads.                              |
| Docker       | Docker                    | Containerization platform for packaging AI applications.                |
| K8s          | Kubernetes                | Orchestration tool for managing Docker containers.                      |
| MLOps        | Machine Learning Operations| Practices to deploy, monitor, and manage ML systems in production.      |

---

## 🧪 Evaluation Metrics

| Abbreviation | Term                          | Description                                                                |
|--------------|-------------------------------|----------------------------------------------------------------------------|
| BLEU         | Bilingual Evaluation Understudy| Measures accuracy of machine-translated text.                              |
| ROUGE        | Recall-Oriented Understudy for Gisting Evaluation | Compares overlap of generated vs. reference summaries.        |
| METEOR       | Metric for Evaluation of Translation with Explicit ORdering | Evaluates machine translation with synonyms and order.   |
| Perplexity   | —                             | Measures how well a language model predicts text. Lower is better.        |
| Accuracy     | —                             | Proportion of correct predictions.                                        |
| F1 Score     | —                             | Harmonic mean of precision and recall. Used in classification tasks.      |

---

## 🧩 AI Teams & Organizations

| Organization             | Focus                                                                |
|--------------------------|----------------------------------------------------------------------|
| OpenAI                   | LLMs (ChatGPT), AGI research.                                        |
| Anthropic                | Claude models, alignment-focused AI.                                |
| Google DeepMind          | AlphaGo, AGI research, robotics, RL.                                 |
| Meta AI (FAIR)           | LLaMA models, open-source NLP and vision research.                   |
| Microsoft Research       | Azure AI, partnerships with OpenAI, responsible AI.                  |
| HuggingFace              | Model sharing, Transformers library.                                 |
| EleutherAI               | GPT-Neo, GPT-J open-source LLMs.                                     |
| Stability AI             | Creator of Stable Diffusion and other generative models.             |
| Together AI              | Open models and decentralized AI compute.                            |
| Llama.cpp                | Efficient CPU-based LLM inference.                                   |
| Ollama                   | Local LLM runner, optimized for Apple Silicon.                       |
| Mistral AI               | High-performance open LLMs (e.g., Mistral 7B).                        |
| xAI                     | Elon Musk’s AGI research initiative.                                 |
| Baidu ERNIE Bot          | LLMs in the Chinese language space.                                  |
| Alibaba Cloud (Qwen)     | Qwen LLM series for general AI use.                                  |
| Tencent HunYuan          | General-purpose AI and LLMs.                                         |
