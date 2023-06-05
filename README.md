# Sniffer

**工具名称：Sniffer - AI起源追溯工具**

Sniffer是一款专为追溯和检测基于AI系统生成的上下文起源而设计的工具。随着大型语言模型（LLMs）的广泛应用，对于上下文是否由AI系统生成的判断变得尤为重要。然而，随着越来越多的公司和机构发布自己的LLMs，追溯这些模型的起源却变得困难。

Sniffer旨在解决这一问题，通过提供一种有效的方法来追踪和检测AI生成的上下文的起源。我们的工具引入了一种全新的算法，利用LLMs之间的对比特征，并提取模型特征来追溯文本的起源。与传统的监督学习方法相比，我们的方法只需要有限的数据，并且可以轻松扩展到追溯新模型的起源。

Sniffer的主要特点包括：

1. **适用广泛**：我们的方法在白盒和黑盒设置下均有效，可以追溯和检测各种LLMs，例如GPT-3模型等。
2. **高效准确**：Sniffer利用对比特征和模型特征的组合，能够快速而准确地确定上下文是否由AI系统生成，从而帮助用户追溯起源。
3. **少量数据需求**：相比于传统的监督学习方法，我们的工具只需要少量的数据就能达到良好的追溯效果，减少了用户数据收集的负担。
4. **开放工具包和基准数据**：我们提供了完整的代码和数据作为工具包和基准，方便研究人员进行AI起源追溯和检测的进一步研究。同时，我们呼吁对LLM提供者进行伦理关注，以保障AI技术的道德使用。

我们通过详细的实验证明了Sniffer的可行性和有效性，并提供了很多宝贵的观察结果。我们相信Sniffer将为AI起源追溯领域的研究和实践提供有力的支持。请随时使用Sniffer工具，并参考我们的代码和数据，探索更多关于AI起源追溯和检测的可能性。如果您有任何问题或建议，请随时联系我们。

## 数据集

我们搜集了一个数据集SnifferBench，用于训练和评测模型分别AI生成样本的能力。

## 代码使用

- 你可以使用backend_api.py部署后端模型，从而获取样本的perplexity list。

```
python backend_api.py --port 6006 --timeout 30000 --debug --model=damo --gpu=3
```

- 我们提供了sniffer分类器的训练代码和测试代码，相关的代码保存在`sniffer`目录下。

  **训练代码**

  - `train.py`：sniffer分类器训练代码。
  - `train_j_neo_divided.py`：将gptj，gptneo当作不同来源，训练sniffer分类器。
  - `train_white_box.py`：结合了gpt3.5 text-003返回的logits的sniffer分类器训练代码。
  - `train_with_semantic.py`：结合了roberta语义向量的分类器训练代码，即结合了sniffer model-wise的特征和roberta semantic-wise的特征。

  **测试代码**

  - `test.py`：对训练好的sniffer分类器进行测试。

  - `test_semantic.py`：对结合了semantic-wise和model-wise特征训练出来的分类器进行测试。

- 我们还提供了基线模型Roberta分类器的相关训练代码，保存在`roberta`目录下。
- 我们还提供了detectGPT消融实验时，用于生成扰动数据的相关代码，保存在detectgpt目录下。

