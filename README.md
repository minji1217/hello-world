# 논문 추천 시스템 파이프라인

> Citation context 기반 논문 추천 시스템 — Retrieval → Graph Expansion → Ranking → Reranking

이 문서는 전체 논문 추천 파이프라인의 데이터 흐름, 후보 수 변화, 모듈별 입출력을 정리합니다.

---

## 1. 전체 흐름

### 1.1 Offline → Online 파이프라인

```text
[Offline]
raw corpus
    → 전처리
    → SPECTER2 embedding
    → FAISS index
    → citation_graph.pkl
    → metadata_db

[Online]
논문 입력
    → Module 2: Retrieval
      → offline_output.json
    → Module 3: Graph Expansion & Feature Engineering
      → output_graph_stage3.json
    → Module 4-1: 1차 Ranking
      → stage4_1_output.json
    → Module 4-2: Cross Encoder (2차 Ranking)
      → final_output.json
```

### 1.2 후보 수 변화

| 단계        | 설명                                        | 후보 수         |
|-------------|---------------------------------------------|-----------------|
| 초기        | FAISS semantic retrieval                    | Top‑3000        |
| Module 2    | paper/context score fusion + bib soft bias | Top‑150         |
| Module 3    | citation graph 확장                         | 약 225    |
| Module 4‑1  | feature 기반 1차 Ranking                    | Top‑30          |
| Module 4‑2  | Cross Encoder 2차 Ranking                   | 최종 Top‑10     |

---

## 2. Module 2 — Retrieval

Semantic retrieval과 bibliography soft bias를 통해 context 기반 후보 Top‑150을 선정합니다.

### 2.1 입력

| 항목              | 설명                                                          |
|-------------------|---------------------------------------------------------------|
| `context_queries` | 인용 위치별 문맥 텍스트 N개 (`query_id`별 1개)               |
| `paper_query`     | 논문 전체를 대표하는 query 1개 (optional)                    |
| `embedding_db`    | `{ paper_id → vector }` 형태의 임베딩 DB                     |
| `faiss_index`     | 사전 구축된 ANN 인덱스 (IndexFlatIP, L2-normalized)          |
| `bib_ids`         | 유저 bibliography paper_id 목록 (optional)                   |

### 2.2 동작 개요

1. Context Embedding  
   - 각 `context_queries`를 SPECTER2에 넣어 context embedding을 생성합니다.

2. FAISS Semantic Retrieval (Top‑3000)  
   - context embedding q를 `faiss_index`에 질의하여 cosine similarity 기준 Top‑3000 후보 논문을 검색합니다.  
   - 검색 시 유사도 값을 각 후보의 `paper_similarity`로 저장합니다.

3. Context Similarity 및 Score Fusion (Top‑1500)  
   - Top‑3000 후보 각각에 대해 context embedding과의 cosine을 다시 계산하여 `context_similarity`를 구합니다.  
   - `paper_similarity`와 `context_similarity`를 가중합하여 통합 점수 `score_paper_context`를 계산합니다.  
   - `score_paper_context` 기준으로 상위 1500편만 남기도록 1차 pruning을 수행합니다.

4. Bibliography Soft Bias 및 Top‑150 선정  
   - Top‑1500 후보에 대해 user bibliography 기반 `bib_score`를 계산합니다.  
   - 최종 점수 `final_similarity = w1 * score_paper_context + w2 * bib_score` 로 정의합니다.  
   - `final_similarity` 기준 상위 150편을 선택하여 다음 단계(그래프 확장 및 랭킹)로 넘깁니다.

### 2.3 출력 — `offline_output.json`

```json
{
  "query_id": "arxiv_xxx_00",
  "target_ids": ["s2_xxx"],
  "context": "...",
  "candidates": [
    {
      "paper_id": "s2_xxx",
      "sim": 0.91,
      "bib_score": 0.74
    }
  ]
}
```

- `sim`: paper/context score fusion 결과  
- `bib_score`: user bibliography 기반 soft bias 점수

---

## 3. Module 3 — Graph Expansion 및 Feature Engineering

Module 2의 Top‑150 후보를 기반으로 citation graph 확장과 feature 계산을 수행합니다.

### 3.1 입력

| 항목                 | 설명                                                                 |
|----------------------|----------------------------------------------------------------------|
| `offline_output.json` | Module 2 출력 (query별 후보 Top‑150)                               |
| `citation_graph.pkl` | `{ "forward": {paper_id → [paper_id]}, "backward": {paper_id → [paper_id]} }` 구조의 citation graph |
| `metadata_db`        | `{ paper_id → { "year": int, "citation_count": int } }`             |
| `embedding_db`       | graph 확장 후보의 `semantic_score` 계산용 임베딩 DB                |

### 3.2 동작 개요

1. Graph Expansion  
   - 각 후보 논문에 대해 `forward`(참조하는 논문), `backward`(인용하는 논문)를 따라 인접 논문을 추가합니다.  
   - semantic 후보와 graph 후보를 합쳐 약 225편 수준으로 후보 풀을 확장합니다.

2. Feature Engineering  
   - 확장된 모든 후보에 대해 임베딩, 메타데이터, 그래프 정보를 기반으로 feature를 계산하고 0~1 범위로 정규화합니다.

### 3.3 Feature 설명

| feature              | 설명                                                                 |
|----------------------|----------------------------------------------------------------------|
| `semantic_score`     | SPECTER2 기반 cosine similarity (context vs. paper)                 |
| `bib_score`          | bibliography 임베딩과의 cosine similarity top‑5 평균                |
| `graph_score`        | citation graph 양방향 연결 강도 (co‑citation, 공통 이웃 수 등 → min‑max 정규화) |
| `citation_count_log` | `log1p(citation_count)`                                              |
| `recency_score`      | `1 / (1 + max(0, 2026 - year))`                                     |
| `source_faiss`       | FAISS semantic retrieval 출처 여부 (`true` / `false`)               |
| `source_graph`       | graph 확장 출처 여부 (`true` / `false`)                             |
| `has_forward`        | references 연결 존재 여부 (`true` / `false`)                        |
| `has_backward`       | cited_by 연결 존재 여부 (`true` / `false`)                          |
| `forward_count`      | 참조하는(reference) 논문 수                                         |
| `backward_count`     | 인용하는(cited_by) 논문 수                                          |
| `graph_available`    | 그래프 정보가 충분히 존재하는지 여부 (`true` / `false`)            |
| `is_target`          | 학습 및 평가 시 정답 인용 목록에 포함 여부 (`true` / `false`)      |

### 3.4 출력 — `output_graph_stage3.json`

```json
{
  "query_id": "arxiv_xxx_00",
  "query_paper_id": "arxiv_xxx",
  "context": "...",
  "target_ids": ["s2_xxx"],
  "candidates": [
    {
      "paper_id": "s2_xxx",
      "semantic_score": 0.685,
      "bib_score": 0.799,
      "graph_score": 0.618,
      "citation_count_log": 4.21,
      "recency_score": 0.5,
      "source_faiss": true,
      "source_graph": false,
      "has_forward": true,
      "has_backward": true,
      "forward_count": 35,
      "backward_count": 129,
      "graph_available": true,
      "is_target": false
    }
  ]
}
```

---

## 4. Module 4‑1 — 1차 Ranking

Graph Expansion까지 확장된 후보(약 225개)를 feature 기반 선형 점수로 랭킹하여 Top‑30으로 압축합니다.

### 4.1 입력

| 항목                      | 설명                                                              |
|---------------------------|-------------------------------------------------------------------|
| `output_graph_stage3.json` | Module 3 출력                                                   |
| 사용 feature              | `semantic_score`, `bib_score`, `graph_score`, `forward_count`, `backward_count`, `graph_available` |
| weight config             | (alpha, beta1, beta2, beta3, beta4, beta5), grid search로 튜닝된 가중치 |

### 4.2 Scoring 수식

각 후보 논문 p에 대해 1차 점수 score_1(p)는 다음과 같이 계산합니다.

$$
\begin{aligned}
score_1(p) = {} & \alpha \cdot \text{semantic\_score} \\
               & + \beta_1 \cdot \text{bib\_score} \\
               & + \beta_2 \cdot \text{graph\_score} \\
               & + \beta_3 \cdot \text{forward\_count} \\
               & + \beta_4 \cdot \text{backward\_count} \\
               & + \beta_5 \cdot \text{graph\_available}
\end{aligned}
$$

- score_1 기준으로 내림차순 정렬하여 Top‑30(또는 30~50) 후보만 남깁니다.  
- 목적: Cross Encoder가 처리해야 할 입력 수를 줄이고 노이즈를 제거하는 빠른 필터링 단계입니다.

### 4.3 출력 — `stage4_1_output.json`

```json
{
  "query_id": "arxiv_xxx_00",
  "context": "...",
  "target_ids": ["s2_xxx"],
  "candidates": [
    {
      "paper_id": "s2_xxx",
      "score1": 0.812
    }
  ]
}
```

225개 내외 후보 → Top‑30 후보로 압축 (Cross Encoder 입력 수 제한 목적).

---

## 5. Module 4‑2 — Cross Encoder (2차 Ranking)

1차 랭킹 결과 Top‑30 후보에 대해 문맥–논문 쌍을 입력으로 하는 Cross Encoder를 적용해 최종 Top‑K 인용 추천을 생성합니다.

### 5.1 입력

| 항목                  | 설명                                              |
|-----------------------|---------------------------------------------------|
| `stage4_1_output.json` | Module 4‑1 출력 (Top‑30 후보)                   |
| `abstract_db`         | `{ paper_id → title + abstract }`                |

- 각 후보에 대해 입력 쌍: [context, title + abstract]

### 5.2 모델

- Cross Encoder: ms-marco-MiniLM-L-6-v2

### 5.3 출력 — `final_output.json`

```json
{
  "query_id": "arxiv_xxx_00",
  "context": "...",
  "target_ids": ["s2_xxx"],
  "candidates": [
    {
      "paper_id": "s2_xxx",
      "score1": 0.812,
      "score2": 0.934
    }
  ]
}
```

- score1: Module 4‑1에서 계산한 1차 선형 랭킹 점수  
- score2: Cross Encoder가 계산한 최종 relevance score  

각 query에 대해 score2 기준으로 정렬하여 최종 Top‑K 후보를 인용 추천 결과로 사용합니다.

---

## 6. 요약

- Module 2: FAISS와 bibliography soft bias로 context 기반 Top‑150 후보 생성  
- Module 3: citation graph 기반 확장 및 feature engineering (semantic, bib, graph, recency 등)  
- Module 4‑1: 선형 결합을 이용한 1차 Ranking, Top‑30 필터링  
- Module 4‑2: Cross Encoder로 문맥–논문 쌍 단위 정교한 2차 Ranking → 최종 Top‑K

이 파이프라인을 통해 대규모 논문 코퍼스에서도 문맥 기반, 개인화, 그래프 정보를 모두 반영한 인용 추천을 효율적으로 수행할 수 있습니다.
```
