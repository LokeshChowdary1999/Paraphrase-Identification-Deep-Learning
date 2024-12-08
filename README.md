# Paraphrase Identification Using Deep Learning

This project explores the task of paraphrase identification using deep learning techniques. Paraphrase identification involves determining whether two sentences convey the same meaning despite differences in wording. The project utilizes multiple neural architectures to evaluate their performance on the Quora Question Pairs dataset.

---

## Objective

The primary goal is to compare and analyze the effectiveness of different deep learning models for identifying semantic equivalence between sentence pairs.

---

## Project Structure

This repository is organized as follows:

- **`NLPFinalProject.ipynb`**: Jupyter Notebook containing the implementation, training, and evaluation of all models. Includes preprocessing, architecture details, and results.
- **`ProjectPresentation.pptx`**: A presentation summarizing the project's objectives, methodology, and findings.
- **`Paraphrase Identification with Deep Learning.pdf`**: Final paper documenting the project’s scope, methodologies, experimental results, and conclusions.
- **`requirements.txt`**: A list of required Python libraries for running the project.
- **`Architecture_Images/`**: Contains architecture diagrams of the models used (FFN, Bi-LSTM GRN, Siamese Network, and CNN).
- **`Results/`**: Contains accuracy and loss values for each model.

---

## Dataset

The project uses the **Quora Question Pairs dataset**, which includes:
- **404,000 question pairs** labeled as either "Duplicate" or "Not Duplicate."
- An 80-20 split for training and testing.

---

## Key Features

1. **Preprocessing**: Text cleaning, tokenization, padding, and embedding using GloVe vectors.
2. **Deep Learning Models**:
   - **Feedforward Neural Network (FFN)**
   - **Bi-LSTM with Gated Relevance Network (GRN)**
   - **Siamese Network**
   - **Convolutional Neural Network (CNN)**
3. **Model Evaluation**:
   - Metrics: Accuracy, Precision, Recall, F1-Score
   - Comprehensive analysis of performance and limitations.

---

## Results

| Model            | Accuracy  | Precision | Recall   | F1-Score |
|-------------------|-----------|-----------|----------|----------|
| Feedforward NN   | 66.85%    | 58.77%    | 36.07%   | 44.70%   |
| Bi-LSTM + GRN    | **71.20%**| **60.58%**| **64.33%**| **62.40%**|
| Siamese Network  | 71.10%    | 65.25%    | 47.51%   | 54.98%   |
| CNN              | 66.50%    | 54.17%    | 63.80%   | 58.59%   |

---

## Future Work
Implement transformer-based models like BERT or GPT for improved performance.
Extend the project to support multilingual datasets for global applications.
Develop ensemble models combining different architectures for robust predictions.

## References
1.Zhou, C., Qiu, C., & Acuna, D. E. (2024). Paraphrase Identification with Deep Learning: A Review of Datasets and Methods.

2.Yin, W., & Schutze, H. (2015). Convolutional Neural Network for Paraphrase Identification. Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT).

3.Lan, W., & Xu, W. (2018). Neural Network Models for Paraphrase Identification, Semantic Textual Similarity, Natural Language Inference, and Question Answering.

4.Peinelt, N., Nguyen, D., & Liakata, M. (2020). Better Early than Late: Fusing Topics with Word Embeddings for Neural Question Paraphrase Identification.

5.Vrbanec, T., & Mestrovic, A. (2021). Corpus-Based Paraphrase Detection Experiments and Review.
