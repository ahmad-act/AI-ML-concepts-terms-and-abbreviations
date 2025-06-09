# AI Capabilities Overview

This table outlines various AI capabilities, their use cases, examples, and technical approaches, organized by domain. The enhanced version includes improved formatting, consistent styling, and additional details for clarity.

| #   | Capability                       | Use Case                                                  | Example                                                   | Sample Modes / Approaches                     |
|-----|----------------------------------|-----------------------------------------------------------|-----------------------------------------------------------|-----------------------------------------------|
| **Multimodal**                                                                                                                           |
| 1   | Audio-Text-to-Text               | Transcribe and summarize meetings using audio and text inputs | Audio of meeting + partial notes → concise summary         | Speech recognition (e.g., Whisper) + text summarization (e.g., BART) |
| 2   | Image-Text-to-Text               | Generate detailed descriptions combining images and text   | Image of a painting + title → descriptive caption          | Vision Transformer (ViT) + GPT-based text generation |
| 3   | Visual Question Answering (VQA)  | Answer questions about visual content                      | Image of a street + "How many cars?" → "Three cars"        | CNN (ResNet) + Transformer-based QA           |
| 4   | Document Question Answering      | Extract specific information from scanned/digital documents | Scanned invoice + "What’s the total?" → "$150.75"         | OCR (Tesseract) + Language Model (BERT)       |
| 5   | Video-Text-to-Text               | Summarize video content with additional text context       | Video clip + prompt → key points summary                  | Video feature extraction (S3D) + NLP (T5)     |
| 6   | Visual Document Retrieval        | Search documents based on visual content                   | Image of a form → related documents                       | Image embedding (CLIP) + similarity search    |
| 7   | Any-to-Any Modal Conversion      | Transform data across modalities (audio, text, image, video) | Voice command → video with subtitles                      | Multimodal embeddings (e.g., MMBT)            |
| **Computer Vision**                                                                                                                      |
| 1   | Depth Estimation                 | Create 3D depth maps from 2D images                        | RGB photo of a room → depth map                           | CNN regression (MiDaS)                       |
| 2   | Image Classification             | Identify objects or scenes in images                       | Image → labeled as "dog" or "car"                         | CNNs (ResNet50, EfficientNet-B7)             |
| 3  | Object Detection                 | Locate and classify objects with bounding boxes            | Image → detects faces, cars with coordinates               | YOLOv8, Faster R-CNN                         |
| 4  | Image Segmentation              | Assign labels to each pixel for detailed region detection   | Image → segments road, cars, pedestrians                  | U-Net, Mask R-CNN                            |
| 5  | Text-to-Image                    | Generate images from textual descriptions                   | Prompt: "A red bird on a tree" → image                    | Stable Diffusion, DALL·E 3                    |
| 6  | Image-to-Text                    | Create captions or descriptions from images                 | Photo → "People hiking in mountains"                      | CNN (VGG) + Transformer decoder (BLIP)        |
| 7  | Image-to-Image                   | Transform images (e.g., style transfer, super-resolution)   | Daytime photo → nighttime version                         | CycleGAN, SRGAN                               |
| 8  | Image-to-Video                   | Generate animated videos from static images                | Portrait → talking avatar video                           | First Order Motion Model, GANs                |
| 9  | Unconditional Image Generation   | Create images without specific input constraints           | Generate abstract art images                              | StyleGAN3, VQ-VAE-2                           |
| 10  | Video Classification             | Categorize video content                                   | Video → labeled as "soccer match"                         | 3D CNNs (I3D), Video Swin Transformer         |
| 11  | Text-to-Video                    | Produce short videos from text prompts                     | Prompt: "Cat playing with ball" → video                   | Video Diffusion Models (Make-A-Video)         |
| 12  | Zero-Shot Image Classification   | Classify images into unseen categories using text cues     | Image + text: "New bird species" → classification         | CLIP, ViT-L-336px                             |
| 13  | Mask Generation                  | Create masks for image segmentation or editing             | Image → background mask for photo editing                 | Mask R-CNN, SAM (Segment Anything Model)      |
| 14  | Zero-Shot Object Detection       | Detect objects not seen during training                    | Image + text: "Unicorn" → bounding box                    | CLIP + Grounding DINO                         |
| 15  | Text-to-3D                       | Generate 3D models from text descriptions                   | Prompt: "Red chair with four legs" → 3D mesh              | DreamFusion, Point-E                          |
| 16  | Image-to-3D                      | Reconstruct 3D models from multiple images                 | Photos of a statue → 3D model                             | COLMAP, NeRF-W                                |
| 17  | Image Feature Extraction         | Extract embeddings for similarity or retrieval tasks       | Image → embedding for visual search                       | ResNet101, CLIP-ViT                           |
| 18  | Keypoint Detection               | Identify human poses or object landmarks                   | Image → joint positions for fitness tracking              | OpenPose, HRNet                               |
| **Natural Language Processing**                                                                                                          |
| 1  | Text Classification              | Categorize text (e.g., spam detection, sentiment analysis) | Email → "spam" or "not spam"                              | BERT, RoBERTa                                 |
| 2  | Token Classification             | Label tokens for NER or POS tagging                        | Text → "person: John, location: Paris"                    | DistilBERT, XLM-R                             |
| 3  | Table Question Answering         | Answer queries over structured tabular data                | Table + "Q1 sales total?" → "$500K"                       | TAPAS, TableFormer                            |
| 4  | Question Answering               | Extract answers from text or knowledge bases               | Query: "Who wrote Hamlet?" → "William Shakespeare"        | T5, UnifiedQA                                 |
| 5  | Zero-Shot Classification         | Classify text without labeled training data                | Feedback → categorized as "urgent" or "routine"           | BART-MNLI, Flan-T5                            |
| 6  | Translation                      | Convert text between languages                             | English text → French translation                         | MarianMT, mBART                               |
| 7  | Summarization                    | Produce concise summaries of long texts                    | News article → 2-sentence summary                         | Pegasus, BART-Large-CNN                       |
| 8  | Feature Extraction               | Generate text embeddings for clustering or retrieval       | Text → embedding for semantic search                      | Sentence-BERT, MiniLM                         |
| 9  | Text Generation                  | Create coherent paragraphs or stories                      | Prompt → short story continuation                         | GPT-4, LLaMA                                  |
| 10  | Text2Text Generation             | Paraphrase, simplify, or rephrase text                     | Complex text → simpler version                            | T5, Parrot                                    |
| 11  | Fill-Mask                        | Predict missing words in sentences                         | "The cat is on the [MASK]." → "mat"                       | RoBERTa, ALBERT                               |
| 12  | Sentence Similarity              | Measure semantic similarity between sentences              | Two questions → similarity score                          | Sentence-BERT, SimCSE                         |
| 13  | Text Ranking                     | Rank documents by relevance to a query                     | Query → ordered list of search results                    | ColBERT, MonoT5                               |
| **Audio**                                                                                                                                |
| 1  | Text-to-Speech (TTS)             | Convert text into natural-sounding speech                  | Text → audio narration for audiobooks                     | Tacotron 2, VITS                              |
| 2  | Text-to-Audio                    | Generate audio effects or sounds from text                 | Prompt: "Sound of thunder" → audio clip                   | AudioLDM, SpecVQGAN                           |
| 3  | Automatic Speech Recognition (ASR) | Transcribe spoken audio into text                        | Podcast audio → text transcript                           | Whisper, Wav2Vec 2.0                          |
| 4  | Audio-to-Audio                   | Enhance or transform audio (e.g., noise reduction)         | Noisy recording → clean audio                             | DEMUCS, Wave-U-Net                            |
| 5  | Audio Classification             | Identify sound events or categories                        | Audio → "dog bark" or "gunshot"                           | YAMNet, VGGish                                |
| 6  | Voice Activity Detection (VAD)   | Detect presence of human speech in audio                   | Audio → speech/non-speech segments                        | WebRTC VAD, Silero VAD                        |
| **Tabular**                                                                                                                              |
| 1  | Tabular Classification           | Classify rows in structured datasets                       | Customer data → predict churn (yes/no)                    | XGBoost, CatBoost                             |
| 2  | Tabular Regression               | Predict continuous values from tabular data                | House features → price prediction                         | LightGBM, Neural Nets (TabNet)                |
| **Time Series**                                                                                                                          |
| 1  | Time Series Forecasting          | Predict future values based on historical time series      | Stock prices → next week’s forecast                       | LSTM, Informer, Prophet                       |
| **Reinforcement Learning**                                                                                                               |
| 1  | Reinforcement Learning (RL)      | Train agents to optimize decisions via rewards             | Robot navigation, game AI (e.g., chess)                   | PPO, DQN, SAC                                 |
| **Robotics**                                                                                                                             |
| 1  | Robotics                         | Enable perception and control for robotic systems          | Drone → autonomous obstacle avoidance                     | RL (DRL) + Computer Vision (YOLO)             |
| **Other**                                                                                                                                |
| 1  | Graph Machine Learning           | Model relationships in graph-structured data               | Protein network → predict interactions                    | GCN, GraphSAGE, GAT                           |

## Notes
- **Domains**: Capabilities are grouped by Multimodal, Computer Vision, NLP, Audio, Tabular, Time Series, Reinforcement Learning, Robotics, and Other.
- **Approaches**: Modern models and frameworks are listed for each capability, reflecting state-of-the-art techniques as of 2025.
- **Use Cases**: Examples are practical and grounded in real-world applications.
- For further details on any capability, refer to specific model documentation or research papers linked to the listed approaches.