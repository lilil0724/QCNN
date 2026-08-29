# Qwen3 與 Ternary Bonsai 的研究方法：可追溯技術報告

> 易讀版訓練流程：[`WEIGHT_MAP_TRAINING_PIPELINE_ZH_TW.html`](WEIGHT_MAP_TRAINING_PIPELINE_ZH_TW.html)

## 摘要

本研究以全精度 `Qwen/Qwen3-1.7B` 與公開的
`prism-ml/Ternary-Bonsai-1.7B-unpacked` 為配對資料，研究「低位元模型的最終權重，
有多少能由原始全精度權重及其局部結構預測」。目前方法包含兩個層次：第一，直接比較
同名、同形狀權重的靜態結構；第二，將 Bonsai 成品權重拆成三值碼與 group scale，訓練
一個由 Qwen3 權重預測 Bonsai 權重的輔助 weight-map，再以權重空間指標及語言模型
perplexity 評估。這不是重新執行 Prism ML 的量化訓練，也尚不能重現其模型生產流程。
本 repo 的 handoff 亦明載，初始比較是靜態 weight-tensor diff，沒有以文字或 activation
進行兩模型逐層表徵比較（`docs/HANDOFF.md`, L10–21）。

最重要的證據界線是：發布方公開了成品的三值表示、每 128 個權重共享一個 FP16 scale，
以及哪些主要矩陣採三值權重；但沒有公開可重現的 PTQ/QAT、蒸餾、資料、optimizer、
learning-rate schedule、STE 或 scale 更新公式。故論文中應把 Bonsai 稱為「由 Qwen3-1.7B
衍生、具有 ternary-g128 成品表示的公開 checkpoint」，而不是把某個常見 BitNet/QAT
配方寫成已知事實（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L13–17, L101–141）。

## 1. 證據規則與術語

本報告只採用 repo 內可追溯的一手材料：執行中的程式碼、保存的實驗文件、模型 metadata
稽核筆記與已記錄的實跑結果。每個敘述分成三類：

1. **發布方明示**：Prism ML／Qwen 的模型卡、白皮書或 config 內容，已由 repo 的 primary-
   source audit 留下來源紀錄。
2. **本 repo 實證**：程式對公開 checkpoint 的結構檢查或實際 GPU run。
3. **研究推論**：由上述結果導出的解讀，不能反向當成發布方訓練配方。

此外，repo 舊文件常使用「ternary-QAT」描述 Bonsai，但較新的來源稽核指出，發布方並未
公開 PTQ 或 QAT 的判定證據。因此本文以「Bonsai 目標 checkpoint」或「三值成品權重」為
主稱呼；只有描述 repo 既有變數名稱或歷史假設時才保留 QAT 一詞
（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L115–141）。

### 1.1 研究狀態總表

| 工作項目 | 狀態 | 可作的論文主張 | 證據 |
|---|---|---|---|
| Qwen3-1.7B 與 Ternary-Bonsai-1.7B 架構、權重配對檢查 | **已完成** | 28 個同架構 blocks；196 個 block projection matrices 可配對 | `docs/BONSAI_QWEN3_1.7B_FINDINGS.md`, L47–55；`docs/MASTER_PLAN_WEIGHT_MAP.md`, L487–495 |
| Bonsai 成品的 ternary code/group scale 結構檢查 | **已完成** | 成品符合 g128；repo inspection 支持 row-contiguous groups | `docs/BONSAI_QWEN3_1.7B_FINDINGS.md`, L57–80；`weight2ternary/data_utils/extract.py`, L122–154 |
| 靜態 sparsity/sign/magnitude/correlation 分析 | **已完成** | 可陳述 28 層 checkpoint comparison 統計 | `docs/BONSAI_QWEN3_1.7B_FINDINGS.md`, L82–110 |
| B0/B1/B2 rule baselines | **已完成** | held-out-layer code accuracy 與 hard-subset 結果 | `docs/MASTER_PLAN_WEIGHT_MAP.md`, L497–519 |
| A0 ContextMLP／A1 GroupConv1D weight-map | **已完成** | 可陳述目前 local-feature family 的小幅增益 | `docs/MASTER_PLAN_WEIGHT_MAP.md`, L521–541 |
| 共同 Bonsai skeleton 的 WikiText-2 PPL | **已完成** | 可比較五種 block replacement modes；不可當成原生 Qwen PPL | `docs/MASTER_PLAN_WEIGHT_MAP.md`, L562–594；`weight2ternary/eval_utils/assemble.py`, L3–24 |
| Prism ML 原始 QAT/PTQ/KD 生產流程重現 | **證據不足／未完成** | 只能把公開 checkpoint 當 observed endpoint | `docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L101–141 |
| 真實資料的逐層 activation/CKA comparison | **僅規劃** | 目前不可主張兩模型內部表徵相似度 | `docs/HANDOFF.md`, L73–104, L123–132 |
| 自行對 Qwen3-0.6B 做多 seed ternary QAT、估 noise ceiling | **僅規劃** | 目前不可主張 62–63% 是跨 seed 的理論上限 | `docs/MASTER_PLAN_WEIGHT_MAP.md`, L413–423, L543–553 |
| 4B/8B、跨 scale／跨 recipe 泛化 | **僅規劃** | 目前核心實證不可外推到其他 scales/recipes | `docs/MASTER_PLAN_WEIGHT_MAP.md`, L558–560 |

**明確更正：本 repo 沒有重現、也沒有重新執行 Ternary Bonsai 的原始 QAT。** 已完成的
工作是對發布 checkpoint 作結構抽取、靜態分析、rule baseline、supervised weight-map
訓練與 block-replacement 功能評估。文件中「QAT ground truth」「QAT checkpoint」等程式
命名只代表低位元 target 的歷史稱呼，不構成發布流程已被確認或重現的證據。

## 2. 模型與成品量化表示

### 2.1 基準模型與架構

基準為 decoder-only `Qwen/Qwen3-1.7B`；目標為
`prism-ml/Ternary-Bonsai-1.7B-unpacked`。Qwen3-1.7B 的官方 config 記錄為
`Qwen3ForCausalLM`，含 28 個 decoder blocks、hidden size 2048、intermediate size
6144、16 個 query heads、8 個 KV heads、head dimension 128、tied input/output
embeddings，以及 vocabulary size 151,936（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`,
L21–35）。本 repo 的實跑確認 Bonsai 亦為相同 model class、28 層、hidden size 2048、
16/8 個 query/KV heads；兩者主要差異是 vocabulary size 151,936 對 151,669，因此
`embed_tokens.weight` 的形狀不相容，而所有 transformer-block 權重皆可同名配對
（`docs/BONSAI_QWEN3_1.7B_FINDINGS.md`, L47–55）。

「架構相同」在本研究中是必要條件，因為方法以參數名稱和 tensor index 作 element-wise
對齊；它不具 permutation invariance，也不適用於不同寬度或不同神經元排列的模型。

模型 ID 中的 `Ternary-` 不可省略。`prism-ml/Bonsai-*` 是 binary `{-1,+1}` family；
`prism-ml/Ternary-Bonsai-*` 才是 ternary `{-1,0,+1}` family。repo 早期曾誤用 binary
checkpoint，導致 near-zero sparsity 與「2× scale gap」等錯誤推論，該批結論已明確作廢
（`docs/BONSAI_QWEN3_1.7B_FINDINGS.md`, L17–43）。所有新實驗除完整記錄 model ID 外，
還必須以 actual zero fraction 與 within-group magnitude structure 作 runtime family check。

### 2.2 可以確定的三值格式

發布方的重建表示可寫成

\[
w_i=s_g t_i, \qquad t_i\in\{-1,0,+1\},
\]

其中每組 128 個 code 共用一個 FP16 scale \(s_g\)，即 ternary g128。三值 code 的
理想資訊量為 \(\log_2 3\approx1.585\) bit/weight；scale 攤提為
\(16/128=0.125\) bit/weight，理想總量約 1.71 bit/weight
（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L37–49）。這和 GGUF `Q2_0` 的實體打包成本
2.125 bit/weight 是不同概念；後者以 34 bytes 儲存 128 個二位元碼與一個 FP16 scale
（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L51–63）。

`unpacked` checkpoint 並非量化前的 latent/master weights；它是把已經三值化的模型展開成
FP16 safetensors，使標準 Hugging Face loader 可直接讀取。因而檔內每個非零值是
`±scale`，而非可用來還原訓練軌跡的連續參數
（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L28–35）。

本 repo 對實際 `model.layers.0.self_attn.q_proj.weight`（2048×2048）的檢查得到 36.6%
exact zeros；每一 output row 含 16 種非零 magnitude，而把 2048 個 input columns 切成
16 個連續的 128-column groups 後，每組只剩一個非零 magnitude。這支持「checkpoint
在 row basis 上以連續 128 columns 分組」的實證解讀（`docs/BONSAI_QWEN3_1.7B_FINDINGS.md`,
L57–72）。但發布方文件只明示 group size 128，未明示 flattening order；所以 row-
contiguous orientation 必須標成**本 repo 的 checkpoint inspection 結果**，不能寫成
發布方規格（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L57–63）。

### 2.3 哪些層被三值化，哪些保留高精度

發布方明示三值權重覆蓋 token embeddings、attention projections、MLP projections 與
language-model head；normalization parameters 與 scale metadata 保留較高精度
（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L65–78）。映射到 Qwen3 模組名稱後，28 個
blocks 中每層的矩陣為：

| 類別 | 每個 block 的參數 | 28 層總數 | 發布成品狀態 | 本 repo weight-map 是否納入 |
|---|---|---:|---|---|
| Attention | `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight` | 112 | ternary g128 | 是 |
| MLP | `gate_proj.weight`, `up_proj.weight`, `down_proj.weight` | 84 | ternary g128 | 是 |
| Embedding | `model.embed_tokens.weight` | 1 | 發布方稱 ternary | 否 |
| LM head | `lm_head.weight`（Qwen 原始 config tied） | 1 個邏輯矩陣 | 發布方稱 ternary | 否 |
| Normalization | block RMSNorm、Q/K norms、final norm | 非矩陣主體 | 高精度 | 否 |
| Scale metadata | 每 128 weights 一個 scale | — | FP16 | 只作預測 target |

上述精確 Qwen 模組名稱是根據 Qwen3 implementation 對發布方類別的展開，並非 Prism ML
逐 tensor 公布的清單；原始 Qwen 的 embedding/head 綁定在 Bonsai 訓練與匯出時如何處理
亦未揭露（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L80–99）。

本 repo 的**研究範圍**比發布成品的量化覆蓋範圍窄。pair extractor 只接受名稱含
`.layers.`、以 `.weight` 結尾、為 2-D，且 base/target 同名同形狀的 tensor；因此自然排除
embeddings、LM head、norms 與所有 shape mismatch（`weight2ternary/data_utils/extract.py`,
L90–115）。實跑共抽取 28 depths × 7 projections = 196 個矩陣
（`docs/MASTER_PLAN_WEIGHT_MAP.md`, L487–495）。所以論文不能寫成「本研究訓練了整個
Bonsai 網路的量化器」；正確說法是「本研究學習 transformer-block 中 196 個 attention/
MLP projection matrices 的 FP-to-ternary mapping」。

### 2.4 目前不能確定的原始量化／訓練方法

目前沒有證據可回答以下「之前怎麼量化 Qwen」的生產細節：PTQ、QAT 或 quantization-
aware continued pretraining；latent/master weight 參數化；STE；訓練中選 code 與更新 scale
的公式；activation 是否量化；資料集、token 數、sequence length、steps、batch size、
optimizer、LR/scheduler、weight decay、gradient clipping、蒸餾 teacher/objective、loss
係數、seed 與 compute。完整未揭露清單見 `docs/BONSAI_PRIMARY_SOURCE_NOTES.md`,
L101–134。

尤其不可用 `onebitllms` 的預設值補空白。repo 的來源稽核顯示該工具使用整個 2-D tensor
共用 absmean scale，且其 module replacement 與 checkpoint utility 排除 embedding、
LM head 或 norms；這與 Bonsai 的 g128 且包含 embedding/head 的發布描述不同
（`docs/BONSAI_PRIMARY_SOURCE_NOTES.md`, L143–175）。因此下文的 absmean/STE 等只代表
**本 repo 的 baseline 或未來比較方法**，不是 Prism ML 的已知做法。

## 3. 靜態權重比較方法

### 3.1 Tensor 配對與層分類

初始分析先載入兩個 causal LM，再枚舉 base model 的所有 2-D named parameters。只有 target
中存在同名且形狀相同的 tensor 才比較；缺少或形狀不符者會明示跳過
（`tools/compare_qat_weights.py`, L184–221）。層別以名稱分為 attention QKVO、MLP
gate/up/down、embedding 與 other（`tools/compare_qat_weights.py`, L82–98）。

### 3.2 統計量定義

令 base tensor 為 \(W\)，target 所得三值碼為 \(T\)，元素總數為 \(N\)，
\(K=\{i:T_i\neq0\}\)，\(Z=\{i:T_i=0\}\)。程式計算：

\[
\text{sparsity}=1-\frac{|K|}{N},
\]

\[
\text{sign-agreement}=\frac{1}{|K|}\sum_{i\in K}
\mathbf{1}[T_i=\operatorname{sign}(W_i)],
\]

\[
R_{\text{kept/zeroed}}=
\frac{|K|^{-1}\sum_{i\in K}|W_i|}{|Z|^{-1}\sum_{i\in Z}|W_i|}.
\]

\(R>1\) 表示 target 非零位置在 base 中平均 magnitude 較大，但它只能支持「與 magnitude
selection 相容」，不能證明訓練演算法就是 magnitude pruning。程式另外沿 row/output
channel 與 column/input channel 計算 zero fraction 及 base L2 norm，再以 Pearson
correlation 衡量 sparsity–norm 關係；標準差近零時回傳 NaN
（`tools/compare_qat_weights.py`, L127–177）。報表先保存 per-layer CSV，再以 layer type
對各層統計做算術平均；這是 **layer-unweighted mean**，不是依 parameter count 加權
（`tools/compare_qat_weights.py`, L228–263）。

需注意歷史工具中的 `ternarize()` 使用
\(\operatorname{clip}(\operatorname{round}(W/\operatorname{mean}|W|),-1,1)\) 的 per-tensor
absmean 規則，而非真實 g128 extractor（`tools/compare_qat_weights.py`, L105–124）。針對已
展開、含 exact-zero cluster 的 Bonsai 成品，直接 `sign()` 才是碼的精確恢復；目前正式
pair extraction 已採後者。故初始靜態結果與後續 code-prediction 結果必須分開解讀。

### 3.3 已記錄的靜態結果

28 層彙總後，attention QKVO 的 sparsity 39.0%、conditional sign agreement 0.866、
kept/zeroed magnitude ratio 1.87、row/column sparsity–norm correlations 為 -0.119/-0.557；
MLP 對應為 40.4%、0.869、1.80、-0.498/-0.620
（`docs/BONSAI_QWEN3_1.7B_FINDINGS.md`, L82–110）。這顯示「保留的非零碼通常延續 base
sign 且位於較大 base magnitude」，但不代表完整三分類 code 可由 magnitude 規則準確
預測，因為 sign agreement 的分母只含 Bonsai 非零位置。

## 4. Ground-truth pair 建立

### 4.1 不實例化模型的抽取

`extract_pair.py` 只下載 safetensors 與 config，透過 index 建立 parameter-to-shard map，
逐 tensor lazy load，不實例化完整 model；這使較大型 pair 可在 CPU 抽取
（`tools/extract_pair.py`, L111–124；`weight2ternary/data_utils/extract.py`, L65–87）。
第一個 matched layer 先經 family assertion，確認實際 tensor structure 符合 binary 或
ternary，而非只相信 repo 名稱；其後每層仍會經 reconstruction check
（`weight2ternary/data_utils/extract.py`, L161–188）。

family check 把 tensor reshape 成 `[out_features, in_features/128, 128]`。每組非零
magnitude 的最大相對 spread 必須 ≤2%；exact-zero fraction ≤0.001 判為 binary，≥0.02
判為 ternary，否則視為 continuous 或提出警告
（`weight2ternary/data_utils/family_check.py`, L26–80）。這些門檻是 repo 的驗證政策，
不是發布方量化演算法。

### 4.2 Code 與 scale 恢復

對每組 target weights \(G\)，code 定為 \(T=\operatorname{sign}(G)\)；scale 是該組
所有非零 magnitude 的 median。全零組 scale 設為 0。若任何非零 magnitude 對 group
median 的最大相對偏差超過 2%，抽取立即失敗
（`weight2ternary/data_utils/extract.py`, L122–154）。median 與 2% tolerance 是為容納
實際 checkpoint 的 bf16-ulp noise：保存紀錄指出極少數（約 \(10^{-4}\)）groups 的最大
within-group relative spread 為 0.73%，其餘多為 bit-exact
（`docs/MASTER_PLAN_WEIGHT_MAP.md`, L514–519）。

每個 projection matrix 產生一個 safetensors shard，保存 base、int8 code、float32
scales、row/column absmean 與 L2 norm；manifest 另存 layer name/type、projection id、
depth、shape、sparsity、tensor absmean 與 reconstruction error
（`weight2ternary/data_utils/extract.py`, L161–228）。

## 5. 非學習式基準

本研究定義三個 ternary code baselines：

- **B0 per-tensor absmean**：整個 tensor 共用 \(a=\operatorname{mean}|W|\)，
  \(\hat T=\operatorname{clip}(\operatorname{round}(W/a),-1,1)\)。
- **B1 per-group absmean**：在每個 row-contiguous 128-column group 內套用相同規則；
  等價於閾值 \(|w_i|\ge0.5\operatorname{mean}_g|W|\) 時保留 sign。
- **B2 tuned group threshold**：
  \(\hat T_i=\operatorname{sign}(W_i)\mathbf{1}[|W_i|\ge
  \tau\operatorname{mean}_g|W|]\)。\(\tau\) 僅在 training layers 上由 0.30 至 0.90、
  間隔 0.025 的 grid search 選擇。

公式與 sweep 實作見 `weight2ternary/eval_utils/baselines.py`, L29–57, L78–94；train-only
tuning 與 full-layer final scoring 見 `tools/run_baselines.py`, L52–104。給定 predicted code
後，baseline scale 是該 group 中 predicted-kept base magnitudes 的平均；若沒有 kept
entry，退回 group absmean（`weight2ternary/eval_utils/baselines.py`, L65–75）。

評估包含 overall accuracy、三類 macro-F1、每類 recall、predicted/true sparsity，以及
**hard subset** accuracy。hard subset 定義為 B1 code 與 Bonsai true code 不同的位置，
用來隔離 trivial baseline 尚未解釋的部分（`weight2ternary/eval_utils/baselines.py`,
L97–135）。

## 6. Weight-map 學習方法

### 6.1 以完整層切分，避免元素洩漏

資料絕不隨機切 element；預設以整個 decoder depth 切分：`depth % 4 == 3` 為 validation，
其餘為 training（`weight2ternary/data_utils/sampler.py`, L13–16, L33–35）。對 28 層而言，
validation depths 是 3、7、11、15、19、23、27，共 7 depths；每個 depth 含七個 projection
matrices，因此是 49 個 validation matrices，其餘 147 個為 training matrices。這個數字
是由公開 split rule 與 196-matrix manifest 推導，不是額外實跑紀錄。

每個 training batch 先隨機選一個 layer，再取 `batch_size` 個隨機 rows，及從 group
boundary 開始的連續 column segment。預設 batch size 64、segment length 512、group size
128；因此每個 row segment 含四組量化 group（`weight2ternary/data_utils/sampler.py`,
L50–73, L134–146；`tools/train_weight_map.py`, L195–223）。validation 對每個 held-out
layer 固定抽 rows（預設 128）並掃過完整 column range
（`weight2ternary/data_utils/sampler.py`, L148–162）。

### 6.2 輸入特徵

每個 weight element 使用 19 個 channels：signed/absolute weight-to-group-absmean ratio、
group 內 magnitude percentile rank、weight-to-group-absmax ratio、group absmean/absmax/std
相對 tensor statistics、row/column absmean、正規化 row/column L2 norm、normalized depth，
以及七種 projection identity one-hot。所有 ratio statistics 取 log 並 clamp 至 [-8,8]；
不編碼 row adjacency（`weight2ternary/data_utils/features.py`, L22–45, L66–105）。此設計的
歸納偏置是：quantization group 與 input-column 結構可能有意義，但 output rows 的排列
本身沒有影像式鄰近語意。

### 6.3 模型架構與 residual parameterization

兩個 PoC 模型皆預測 B1 baseline 上的 **decision-space residual**：code head 輸出加到
margin=4 的 baseline one-hot logits；scale head 輸出加到 `log(baseline_scale)`。
兩個 output heads 均 zero-initialized，因此未訓練模型精確重現 B1
（`weight2ternary/model_utils/build_model.py`, L1–10, L28–58, L118–125）。

- **A0 ContextMLP**：三層 kernel-size-1 Conv1d + GELU，實質為逐 element MLP，預設 hidden
  size 128，不使用鄰近 columns（`weight2ternary/model_utils/build_model.py`, L61–75）。
- **A1 GroupConv1D**：kernel size 9 stem，接 dilation 1、4、16 的 residual Conv1d blocks，
  僅沿同一 row 的 column/group axis 建模（`weight2ternary/model_utils/build_model.py`,
  L78–106）。

code loss 是三類 `{-1,0,+1}` cross-entropy；scale loss 是只在 true scale >0 的 groups 上
計算 log-scale Huber loss：

\[
\mathcal L=\mathcal L_{CE}+\lambda_s\mathcal L_{Huber}
(\log\hat s,\log s), \qquad \lambda_s=0.1\;\text{(default)}.
\]

可選 `hard_weight` 對 B1 錯誤位置加權，但正式校正 runs 使用 1.0；loss 實作見
`weight2ternary/model_utils/losses.py`, L19–50。預設訓練為 20 epochs、每 epoch 500 steps、
AdamW、learning rate 3e-4、weight decay 0.01、cosine annealing、seed 0；每 epoch 依
validation overall code accuracy 選最佳 checkpoint
（`tools/train_weight_map.py`, L101–180, L195–223）。這些是**本 repo weight-map 的超參數**，
不是 Bonsai 原模型生產的超參數。

## 7. 功能性模型組裝與 perplexity

所有功能性 variants 都以同一個真實 Bonsai checkpoint 作 skeleton，固定其 tokenizer、
embeddings、norms 與 LM head，只替換 manifest 中的 196 個 transformer-block linear
weights，以避免 vocabulary mismatch 與非 block 元件干擾比較
（`weight2ternary/eval_utils/assemble.py`, L3–24, L106–149）。五個組裝模式為：

| 模式 | block weight |
|---|---|
| `fp` | Qwen3 base block weights |
| `oracle` | 真實 Bonsai code × 真實 recovered scale |
| `naive_b0` | Qwen3 的 per-tensor absmean code × kept-magnitude mean scale |
| `naive_b1` | Qwen3 的 per-group-128 absmean code × group baseline scale |
| `predicted` | weight-map 預測 code × 預測 group scale |

精確組裝公式見 `weight2ternary/eval_utils/assemble.py`, L80–103。這個控制設計適合比較
block mapping，但 `fp` 並不是原封不動的官方 Qwen3 模型：它仍使用 Bonsai skeleton 的
embedding/head/tokenizer。因此 `fp` PPL 應解讀為「Qwen3 blocks 放入 Bonsai 非 block
元件後的 hybrid control」。

perplexity 使用 WikiText-2 raw test split，先串接非空文本，再切成不重疊的 2048-token
windows；對 shifted targets 累積 token-weighted mean NLL，最後
\(\mathrm{PPL}=\exp(\mathrm{NLL})\)。CLI 預設最多 200 windows、batch size 4
（`tools/eval_perplexity.py`, L33–63, L66–77）。保存的實跑實際使用 145×2048 tokens，
得到 `fp` 27.0、`oracle` 17.7、`naive_b0` 349,604、`naive_b1` 879,504、`predicted`
145,227（`docs/MASTER_PLAN_WEIGHT_MAP.md`, L562–590）。結果支持 predicted mapping 優於
兩個 naive controls，但所有 calibration-free conversions 仍比 oracle 差數個數量級，
不能宣稱已得到可用的 zero-shot ternary language model。

## 8. 目前實驗結果的正確解讀

### 8.1 Gate G1：完整三分類問題比 conditional sign agreement 困難

在 held-out layers 上，B0/B1 overall accuracy 都是 0.606；B2 的 train-tuned
\(\tau=0.775\) 達 0.618。B1 hard fraction 為 0.394，而 B2 在該 subset 的 accuracy 為
0.193（`docs/MASTER_PLAN_WEIGHT_MAP.md`, L497–512）。因此「Bonsai kept positions 中約
87% sign 一致」不能改寫成「可以由 Qwen3 直接還原約 87% 三值碼」；完整 code agreement
只有約 61%。

### 8.2 Gate G2：局部 weight-map 的增益很小

使用 plain CE 並依 overall validation accuracy 選模後，A0/A1 overall accuracy 為
0.623/0.624，hard accuracy 0.187/0.194，scale relative error 0.128/0.127；最佳規則 B2
為 0.618/0.193/0.193。亦即學習模型只提升約 0.6 percentage point，A1 的 column-axis
convolution 幾乎沒有超越逐 element A0（`docs/MASTER_PLAN_WEIGHT_MAP.md`, L521–541）。
這支持「在目前特徵族與 layer split 下，局部可預測訊號有限」；它**不等於**證明 Bonsai
訓練結果本質隨機，也不能外推到全局 attention context、activation-aware 方法、其他模型
scale 或其他量化 recipe。

早期 `hard_weight=4` runs 雖達 hard accuracy 約 0.73，卻使 overall accuracy 降至 0.45，
因此不能當作較佳模型；這也是正式預設改為 `hard_weight=1` 與 overall-accuracy selection
的原因（`tools/train_weight_map.py`, L13–17）。

## 9. 有效性限制與論文揭露事項

1. **Bonsai 生產方法不可重現。** 研究只有成品 checkpoint，缺少原始 QAT/PTQ/KD 配方；
   本文只能分析 endpoint，不能作 optimizer、資料或 STE 層級的因果歸因。
2. **單一 vendor、單一公開成品。** 目前核心結果來自 1.7B pair，無同 recipe 多 seed runs，
   因而無法估計量化訓練的 noise ceiling。
3. **研究只涵蓋 block projections。** 發布方雖稱 embedding/head 也三值化，weight-map
   extractor 與 assembly 沒有學習或替換它們；功能性模型以 Bonsai 非 block 元件固定。
4. **同架構 element-wise 對齊假設。** 方法不處理 neuron/head permutation 或 shape 改變；
   兩模型 vocab mismatch 也使 embedding 不能直接配對。
5. **靜態統計不等於行為。** sparsity、sign agreement、magnitude ratio 與 correlations 不
   衡量 activation 或 downstream representation；handoff 明載尚未進行兩模型的 real-data
   activation comparison（`docs/HANDOFF.md`, L19–21, L73–104）。
6. **局部特徵結論有範圍。** 目前 ceiling 僅針對 row segment、group/row/column statistics
   與 projection/depth conditioning；不能寫成任何模型都無法預測 Bonsai code。
7. **結果可追溯性受限。** repo 文件記錄實跑數值及遠端 results 路徑，但本工作樹沒有納入
   `results/` raw CSV/checkpoints；論文定稿前應從保存環境匯入不可變的 metrics、CLI args、
   git commit、model revisions/checksums 與硬體資訊。現有文件只記錄 196 layers、serials
   1–4 及當時 logs/CSVs 位置（`docs/MASTER_PLAN_WEIGHT_MAP.md`, L487–495）。
8. **軟體驗證不等於模型結論。** synthetic tests 驗證 code/scale round-trip、baseline
   equivalence 與模型可學性，但不能替代真實 checkpoint/GPU run；extractor 對 continuous
   tensor 的拒絕與 synthetic round-trip 見 `tools/extract_pair.py`, L49–78。

## 10. 可直接用於論文 Methods 的建議文字

> We paired the public full-precision Qwen3-1.7B checkpoint with Prism ML's
> Ternary-Bonsai-1.7B-unpacked endpoint. The two checkpoints share the Qwen3 decoder
> architecture, enabling name- and index-aligned comparison of the seven attention and MLP
> projection matrices in each of 28 decoder blocks (196 matrices total). We recovered target
> ternary codes by elementwise sign and recovered one scale per row-contiguous group of 128
> input-channel weights as the median nonzero magnitude, rejecting any tensor whose nonzero
> magnitudes deviated by more than 2% from this group-scaled structure. Embeddings, the language
> model head, and normalization parameters were excluded from weight-map training.
>
> Layers, rather than individual weights, were split between training and validation; decoder
> depths satisfying depth modulo four equals three were held out. We compared per-tensor
> absmean, per-group absmean, and a train-tuned group-threshold baseline with two residual
> predictors: a per-element context MLP and a column-axis dilated 1-D convolutional network.
> Both models predicted ternary-code residual logits and log-scale residuals over the per-group
> absmean baseline, and were trained using code cross-entropy plus log-scale Huber loss. We
> evaluated overall and hard-subset code accuracy, macro-F1, scale relative error, and
> WikiText-2 perplexity after replacing only transformer-block projection weights in a common
> Bonsai model skeleton.
>
> Prism ML discloses the endpoint representation (ternary codes with one FP16 scale per 128
> weights) but not a reproducible training, QAT, or distillation recipe. We therefore treat
> Ternary Bonsai as an observed target checkpoint and make no assumption about its optimizer,
> data, gradient estimator, teacher, or training objective.

以上文字需搭配第 9 節限制，且在正式投稿版本加入可下載的 raw results、checkpoint revision
與完整執行命令，才能達到可重現性要求。
