"""
Sanity check: dung chinh noi dung 1 passage doc lam cau truy van,
kiem tra he thong co tu tim lai duoc chinh no (rank-1) khong.
Chay tu thu muc goc repo ADAPT.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "ReAct")
import local_wikienv
import torch

embedder_name = "dpr-ctx_encoder-single-nq-base"
trigger_sequence = "wondered ##, efficient calls ."

env = local_wikienv.WikiEnv(embedder_name, trigger_sequence, knn=1)

print("\n=== THONG TIN CO BAN ===")
print("Tong so entry trong database:", len(env.database))
print("Tong so embedding_id:", len(env.embedding_id))
print("db_embeddings shape:", env.db_embeddings.shape)
print("poison_ids:", env.poison_ids)

poison_id = sorted(env.poison_ids)[0]
poison_content = env.database[poison_id]["content"]
print(f"\n=== TEST VOI POISON_ID={poison_id} ===")
print("Noi dung passage doc (100 ky tu dau):", poison_content[:100])

tokenized = env.embedding_tokenizer(poison_content, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
input_ids = tokenized["input_ids"].to(local_wikienv.DEVICE)
attention_mask = tokenized["attention_mask"].to(local_wikienv.DEVICE)

with torch.no_grad():
    query_embedding = env.embedding_model(input_ids, attention_mask).pooler_output

query_embedding = query_embedding.to(local_wikienv.DEVICE)
scores = torch.nn.functional.cosine_similarity(query_embedding, env.db_embeddings)
sorted_indices = torch.argsort(scores, descending=True)

top1_index = sorted_indices[0].item()
top1_id = env.embedding_id[top1_index]
top1_score = scores[top1_index].item()

rank_of_self = (env.embedding_id.index(poison_id) if poison_id in env.embedding_id else None)
if rank_of_self is not None:
    self_score = scores[rank_of_self].item()
    self_rank_position = (sorted_indices == rank_of_self).nonzero(as_tuple=True)[0].item()
else:
    self_score = None
    self_rank_position = None

print(f"\n=== KET QUA ===")
print(f"Top-1 retrieved id: {top1_id} (score={top1_score:.4f})")
print(f"Top-1 co phai chinh poison_id={poison_id} khong: {top1_id == poison_id}")
print(f"Diem similarity cua CHINH poison_id voi CHINH no: {self_score:.4f}" if self_score else "KHONG TIM THAY poison_id trong embedding_id!")
print(f"Vi tri xep hang thuc su cua chinh no trong toan bo {len(scores)} entry: {self_rank_position}")