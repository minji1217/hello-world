# 논문 추천 시스템 파이프라인

> Citation context 기반 논문 추천 시스템  
> Retrieval → Graph Expansion → Ranking → Reranking

이 문서는 전체 논문 추천 파이프라인의 데이터 흐름, 후보 수 변화, 모듈별 입출력을 정리합니다.

---

# 1. 전체 흐름

## 1.1 Offline → Online 파이프라인

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

---

## 1.2 후보 수 변화

| 단계 | 설명 | 후보 수 |
|---|---|---|
| 초기 | FAISS semantic retrieval | Top-3000 |
| Module 2 | paper/context score fusion + bib soft bias | Top-150 |
| Module 3 | citation graph 확장 | 약 225 |
| Module 4-1 | feature 기반 1차 Ranking | Top-30 |
| Module 4-2 | Cross Encoder 2차 Ranking | 최종 Top-10 |

---

# 2. Module 2 — Retrieval

Semantic retrieval과 bibliography soft bias를 통해 context 기반 후보 Top-150을 선정합니다.

---

## 2.1 입력

| 항목 | 설명 |
|---|---|
| `context_queries` | 인용 위치별 문맥 텍스트 N개 (`query_id`별 1개) |
| `paper_query` | 논문 전체를 대표하는 query 1개 (optional) |
| `embedding_db` | `{ paper_id → vector }` 형태의 임베딩 DB |
| `faiss_index` | 사전 구축된 ANN 인덱스 (`IndexFlatIP`, L2-normalized) |
| `bib_ids` | user bibliography `paper_id` 목록 (optional) |

---

## 2.2 동작 개요

### 1) Context Embedding

- 각 `context_queries`를 SPECTER2에 입력하여 context embedding 생성

### 2) FAISS Semantic Retrieval (Top-3000)

- context embedding \( q \)를 `faiss_index`에 질의하여 cosine similarity 기준 Top-3000 후보 논문 검색
- 검색된 similarity 값을 `paper_similarity`로 저장

### 3) Context Similarity 및 Score Fusion (Top-1500)

- Top-3000 후보 각각에 대해 context embedding과의 cosine similarity를 다시 계산하여 `context_similarity` 산출
- `paper_similarity`와 `context_similarity`를 가중합하여 통합 점수 `score_paper_context` 계산

```math
score_{\text{paper\_context}}
=
w_a \cdot paper\_similarity
+
w_b \cdot context\_similarity
```

- `score_paper_context` 기준 상위 1500편만 유지하도록 pruning 수행

### 4) Bibliography Soft Bias 및 Top-150 선정

- Top-1500 후보에 대해 user bibliography 기반 `bib_score` 계산
- 최종 점수는 다음과 같이 정의

```math
final\_similarity
=
w_1 \cdot score_{\text{paper\_context}}
+
w_2 \cdot bib\_score
```

- `final_similarity` 기준 Top-150 후보 선정

---

## 2.3 출력 — `offline_output.json`

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

### 필드 설명

- `sim`: paper/context score fusion 결과
- `bib_score`: bibliography 기반 soft bias 점수

---

# 3. Module 3 — Graph Expansion 및 Feature Engineering

Module 2의 Top-150 후보를 기반으로 citation graph 확장 및 feature 계산을 수행합니다.

---

## 3.1 입력

| 항목 | 설명 |
|---|---|
| `offline_output.json` | Module 2 출력 (query별 Top-150 후보) |
| `citation_graph.pkl` | `{ "forward": {paper_id → [paper_id]}, "backward": {paper_id → [paper_id]} }` 형태의 citation graph |
| `metadata_db` | `{ paper_id → { "year": int, "citation_count": int } }` |
| `embedding_db` | graph 확장 후보의 `semantic_score` 계산용 임베딩 DB |

---

## 3.2 동작 개요

### 1) Graph Expansion

- 각 후보 논문에 대해:
  - `forward`: references
  - `backward`: cited-by

를 따라 인접 논문을 추가

- semantic 후보 + graph 후보를 병합하여 약 225편 수준의 후보 풀 생성

### 2) Feature Engineering

- 확장된 후보들에 대해:
  - embedding
  - metadata
  - graph structure

기반 feature 계산 및 정규화 수행

---

## 3.3 Feature 설명

| feature | 설명 |
|---|---|
| `semantic_score` | SPECTER2 cosine similarity (context vs. paper) |
| `bib_score` | bibliography embedding cosine similarity Top-5 평균 |
| `graph_score` | co-citation, 공통 이웃 수 기반 graph strength |
| `citation_count_log` | '''math \( \log(1 + citation\_count) \)''' |
| `recency_score` | 최신성 기반 점수 |
| `source_faiss` | semantic retrieval 출처 여부 |
| `source_graph` | graph expansion 출처 여부 |
| `has_forward` | references 존재 여부 |
| `has_backward` | cited-by 존재 여부 |
| `forward_count` | references 개수 |
| `backward_count` | cited-by 개수 |
| `graph_available` | 충분한 graph 정보 존재 여부 |
| `is_target` | 정답 citation 포함 여부 |

---

### Recency Score

```math
recency\_score
=
\frac{1}{1 + \max(0, 2026 - year)}
```

---

## 3.4 출력 — `output_graph_stage3.json`

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

# 4. Module 4-1 — 1차 Ranking

Graph Expansion 이후 확장된 후보(약 225개)를 feature 기반 선형 결합으로 랭킹하여 Top-30으로 압축합니다.

---

## 4.1 입력

| 항목 | 설명 |
|---|---|
| `output_graph_stage3.json` | Module 3 출력 |
| 사용 feature | `semantic_score`, `bib_score`, `graph_score`, `forward_count`, `backward_count`, `graph_available` |
| weight config | Grid Search로 튜닝된 가중치 |

---

## 4.2 Scoring 수식

각 후보 논문 \( p \)에 대해:

```math
\begin{aligned}
score_1(p)
=
{} &
\alpha \cdot semantic\_score \\
& + \beta_1 \cdot bib\_score \\
& + \beta_2 \cdot graph\_score \\
& + \beta_3 \cdot forward\_count \\
& + \beta_4 \cdot backward\_count \\
& + \beta_5 \cdot graph\_available
\end{aligned}
```

- `score_1` 기준 내림차순 정렬
- Top-30 ~ Top-50만 유지
- 목적:
  - Cross Encoder 입력 수 감소
  - noise candidate 제거

---

## 4.3 출력 — `stage4_1_output.json`

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

> 약 225개 후보 → Top-30 압축

---

# 5. Module 4-2 — Cross Encoder (2차 Ranking)

1차 랭킹 결과 Top-30 후보에 대해 Cross Encoder를 적용하여 최종 citation recommendation을 생성합니다.

---

## 5.1 입력

| 항목 | 설명 |
|---|---|
| `stage4_1_output.json` | Module 4-1 출력 |
| `abstract_db` | `{ paper_id → title + abstract }` |

각 후보 입력 pair:

```text
[context, title + abstract]
```

---

## 5.2 모델

- Cross Encoder: `ms-marco-MiniLM-L-6-v2`

---

## 5.3 출력 — `final_output.json`

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

### 필드 설명

- `score1`: Module 4-1 선형 ranking score
- `score2`: Cross Encoder relevance score

각 query에 대해 `score2` 기준으로 정렬하여 최종 Top-10 recommendation 생성

---

# 6. 요약

- **Module 2**
  - FAISS + bibliography soft bias 기반 retrieval
  - context-aware Top-150 후보 생성

- **Module 3**
  - citation graph expansion
  - semantic / graph / metadata feature engineering

- **Module 4-1**
  - 선형 결합 기반 1차 ranking
  - Top-30 filtering

- **Module 4-2**
  - Cross Encoder 기반 정교한 reranking
  - 최종 Top-10 citation recommendation 생성

---

# 핵심 특징

- Semantic retrieval + graph expansion 결합
- Context-aware citation recommendation
- Personalized bibliography soft bias 반영
- Multi-stage ranking 구조를 통한 효율적 reranking
- 대규모 논문 코퍼스 환경에서도 scalable하게 동작 가능
