# Stage History

本文件記錄專案研究階段的目標、完成條件與交接狀態。真實 checkpoint、GPU、訓練及評估
run 的完整參數與原始結果仍記錄於 [`實驗紀錄.md`](實驗紀錄.md)。每次進入新階段時，
在文件最下方新增一節；不要改寫已完成階段的原始紀錄，修正內容以 erratum 補充。

## Stage 1：Input feature 分析與 ablation 基礎建設

- 開始日期：2026-09-07
- 狀態：進行中
- 研究問題：目前 19-channel、只由 base weights 產生的 input features，哪些能為 B1
  per-group absmean baseline 提供可泛化的 residual correction 訊號？
- 主要模型：ContextMLP 與 GroupConv1D。

### 進入本階段時的基準

- ContextMLP serial 9：validation accuracy `0.6229`，full predicted NLL `12.928100`。
- GroupConv1D serial 10：validation accuracy `0.6243`，full predicted NLL `11.310417`。
- 兩種架構的 accuracy 差距很小，但功能 NLL 差距明顯，因此 feature 判斷不能只依賴
  overall code accuracy。

### 已完成

- 將 19 個 feature channels 整理為 element、group、row、column、depth 與 projection
  groups，並建立 `full`、控制組及 drop-feature ablation sets。
- 所有 ablation 維持固定 19-channel model shape，以 feature mask 控制輸入，避免參數量
  改變干擾比較。
- checkpoint 保存 feature set、feature names、mask、selection metric 與 context halo；舊
  checkpoint 預設為 `full` 並維持可載入。
- GroupConv1D 使用 group-aligned context halo，使 segment training／validation 的中央預測
  與 full-row inference 具有一致鄰近資訊。
- 修正 validation iterator：`in_features` 不是 `segment_len` 整數倍時，最後一個合法、
  group-aligned 的短 segment 也會納入評估。
- feature mask、舊 checkpoint、halo/full-context 一致性與 validation 完整覆蓋的 synthetic
  tests 已通過；尚未執行本階段的真實 GPU ablation runs。

### 下一步與完成條件

1. 產生 feature 分布、冗餘、clamp saturation，以及相對 B1 error/correction target 的靜態
   分析報告。
2. 在 ContextMLP 與 GroupConv1D 上完成所有預定 feature sets 的 seed-0 screening。
3. 對 Full、影響最大與影響最小的 ablation 補 seeds 1、2，使用 paired deltas 判讀結果。
4. 以 validation code、hard repair/easy damage、scale error 與 joint weight reconstruction
   metrics 比較各 feature set。
5. 對入選設定執行固定 WikiText-2 protocol 的 NLL/PPL 驗證，並將所有成功、失敗及重試
   run 追加至 `實驗紀錄.md`。

Stage 1 在上述分析完成、結果具備 multi-seed 支持，且能提出每個 feature group 的保留、
移除或進一步拆解建議後結束。
