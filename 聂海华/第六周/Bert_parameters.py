from transformers import BertModel

bert_path = BertModel.from_pretrained(r"E:\badouai\bert-base-chinese", return_dict=False)

hidden_size = 768  # 隐藏层大小
vocab_size = 21128  # 词汇表大小
max_sequence_length = 512  # 最大句子长度
type_vocab_size = 2  # 类型词汇表大小，分别表示[CLS]和[SEP]两个特殊标记。（最大句子个数）

# embedding过程参数，vocab_size * hidden_size词向量矩阵， type_vocab_size * hidden_size类型向量矩阵，max_sequence_length *
# hidden_size位置向量矩阵，2 * hidden_size分别表示嵌入层归一化层的权重和偏置。layer_norm_weights + layer_norm_biases表示两个hidden_size大小的向量。
# embedding = word_embeddings + type_embeddings + position_embeddings + layer_norm_weights + layer_norm_biases
embedding = vocab_size * hidden_size + type_vocab_size * hidden_size + max_sequence_length * hidden_size + 2 * hidden_size

# q, k, v
self_attention = (hidden_size * hidden_size + hidden_size) * 3

# self_attention过程的参数,q, k, v分别表示query, key, value的权重和偏置，attention_output_weight,
# attention_output_bias表示输出层的权重和偏置。
# q_w. k_w, v_w形状都是hidden_size * hidden_size, q_b, k_b, v_b形状都是hidden_size
# self_attention_out = q + k + v  + attention_output_weight + attention_output_bias
# output=Liner(Attention(Q,K,V))
self_attention_out = (hidden_size * hidden_size + hidden_size) * 3 + hidden_size * 2

# feed_forward过程的参数，intermediate_weight, intermediate_bias, output_weight, output_bias分别表示前馈神经网络中间层和输出层的权重和偏置。
# output=Liner(gelu(Liner(x))) # 激活层是layer_norm层
# 两个线性层一个激活层
feed_forward = hidden_size * hidden_size + hidden_size + hidden_size + hidden_size + hidden_size * hidden_size + hidden_size

# pooler层
# pooler.dense.weight：hidden_size * hidden_size
# pooler.dense.bias：hidden_size
pooler = hidden_size * hidden_size + hidden_size

# 模型总参数 = embedding + self_attention + self_attention_out + feed_forward + pooler
model_all_params = embedding + self_attention + self_attention_out + feed_forward + pooler
print("模型实际参数个数为%d" % sum(p.numel() for p in bert_path.parameters()))
print("diy计算参数个数为%d" % model_all_params)