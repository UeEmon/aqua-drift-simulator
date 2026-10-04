# 要件定義と実装トレーサビリティ

入力された全条件を最終指定に従って整理し、各項目の実装箇所と検証方法を対応付けます。
「未指定」の項目は、採用した既定値と変更方法を明記しています。

凡例：✅ 実装・検証済み ／ 🔧 実装済み（パラメータで変更可）／ 📄 文書・解析ツールとして提供

## 1. 目的・検討対象

| # | 要件 | 状態 | 実装 / 根拠 | 検証 |
|---|---|---|---|---|
| 1.1 | カルマンフィルタの解説、およびそれ以外の候補 | 📄 | `docs/estimation-methods.md` | — |
| 1.2 | 対象：潮流による外力を受ける潜没目標の航跡処理 | ✅ | `physics.advance_target`（対水運動＋流速）、`estimation/` | `test_doppler_only_track_converges...` |
| 1.3 | 空間：3次元 | ✅ | 状態 [東, 北, 深度, 速度3成分, 偏り]、観測者深度・斜距離は3次元 | 同上（深度誤差・σを出力） |
| 1.4 | 入力条件を反映したシミュレーター（代替版ではない） | ✅ | Docker 構成一式（本リポジトリ） | `docker compose config` |

## 2. 目標の運動

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 2.1 | 定針・定速を基準 | ✅ | `advance_target`：指令値に達した後は一定 | `test_target_uses_rate_limited_changes` |
| 2.2 | 対水速力・針路・深度をパラメータで変更 | ✅ | GIS「目標運動」フォーム / `PUT /api/config` の `target.*` | 同上 |
| 2.3 | 変化率：速力 kt/秒、深度 Ft/秒 | ✅ | `speed_rate_kt_per_sec`, `depth_rate_ft_per_sec` | 同上 |
| 2.4 | 針路変化率の単位（未指定） | 🔧 | **度/秒** を採用（`hdg_rate_deg_per_sec`、既定 1.0） | 同上 |
| 2.6 | **目標の初期配置・初期状態を設定するタブ**（初期緯度経度・深度・HDG・対水速力） | ✅ | GIS「目標設定」タブ → `target.initial_*` ＋ `POST /api/reset`（再スタート、地図ダブルクリックで座標入力） | `test_initial_target_state_is_configurable` |
| 2.5 | 推定側の変針・変速への追従 | ✅ | 粒子版 IMM（変針混合）＋ resample-move の窓短縮 | `test_track_recovers_after_maneuver` |

## 3. 外力・流速場

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 3.1 | 一定期間・範囲では線形と仮定（流速場を線形近似） | ✅ | 真値：`current_at`（v = a + G(p − p_ref)）／推定：`CurrentFieldEstimator.fit` | `test_current_field_affine_gradient`, `test_current_field_fit_recovers_affine_field` |
| 3.2 | 探知範囲内の観測者間で係数は一定 | ✅ | 推定位置から 2·R_max 以内の観測者で単一係数を推定 | — |
| 3.3 | 外力判定は観測者位置情報を使用 | ✅ | 観測者の時刻・位置・深度の差分から流速サンプル | 同上 |
| 3.4 | 観測者は周囲の水と同速度で漂流 | ✅ | `advance_observer`（流速のみで移動） | `test_observer_drifts_only_with_current` |
| 3.5 | 期間：探知可能距離 ÷ 目標速力 = 探知可能時間 | ✅ | `TrackingEngine.detectable_time_s()`（推定速力を使用） | GIS「流速場」に期間表示 |
| 3.6 | 流速場の時間更新規則（未指定） | 🔧 | 真値は**時間一定**（パラメータ変更時のみ更新）。推定は 5 秒ごとに探知可能時間窓で再推定 | — |

## 4. 観測者

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 4.1 | 外力のみで移動 | ✅ | `advance_observer` | 上記 |
| 4.2 | 水平位置・深度・時刻は誤差なし | ✅ | 観測に厳密な `observer_position` と `tick` を付与 | — |
| 4.3 | 観測者深度を計算に加味 | ✅ | 斜距離・視線ベクトルはすべて3次元、既定配置で深度を交互に変化 | — |
| 4.4 | 全観測者が同じ対象・同じ音源成分を観測 | ✅ | 単一音源周波数・単一目標 | — |
| 4.5 | 観測者数 1〜100 | ✅ | `observer_limit`（1..100）、観測者は1コンテナ1観測者で `--scale` | `test_fifo_evicts_oldest_and_retains_archive` |
| 4.6 | 上限超過時は最古から削除、観測履歴は保持 | ✅ | `SimulationState.set_observer`（FIFO）、DB `simulation_events` は追記のみ | 同上 |
| 4.7 | 各観測者の観測時間は最大3時間 | ✅ | `max_observation_seconds = 10800`、超過で 410 → コンテナ終了 | `test_observer_expires_after_three_hours` |
| 4.8 | **初期観測者は目標を囲む4点** | ✅ | 既定パターン `surround`：目標初期位置から半径 3000 YD、HDG±45°/135° の4点（深度は 200/350/500 Ft 交互）。5台目以降は外側のリング。再スタート時は新しい初期位置の周囲に再配置（選択可） | `test_default_observers_surround_target`, `test_reset_replaces_default_observers_around_new_target` |
| 4.9 | 観測者の追加配置 | 🔧 | GIS から座標予約、環境変数 `OBSERVER_LAT/LON/DEPTH_FT`、他パターン（grid/line/ring/random） | `test_observer_placement_queue` |

## 5. 観測方式と方位情報

| # | 要件 | 状態 | 実装 |
|---|---|---|---|
| 5.1 | ①位置 ②距離・方位 ③方位のみ の差と組合せの比較 | 📄 | `aqua_drift.analysis.compare_modes`、結果は `docs/observation-mode-comparison.md` |
| 5.2 | 方位：水平・真北基準・15秒・同期・σ=15°・正規・平均0・時間/観測者間独立・範囲内のみ | ✅ | `BearingConfig`（既定 σ 15°, 15 s）、doppler コンテナで生成 | 
| 5.3 | **最新指定：ベアリング（方位）情報も活用する**（以前の「使用しない」を変更） | ✅ | 粒子フィルタの逐次更新・resample-move の窓尤度・LM 残差に方位尤度（ラップした角度差／σ）を追加。`estimator.use_bearing` で切替可（ドップラーのみとの比較用） |
| 5.4 | 「位置が直接得られる」方式は実行系の入力に採用しない | ✅ | 比較対象としてのみ実装 |

検証：`test_bearing_is_horizontal_true_bearing_at_interval`（真方位・σ・間隔）、
`test_bearings_resolve_collinear_mirror_ambiguity`（一直線配置の鏡像解を方位で解消）、
`test_default_surround_with_bearings_tracks_from_start`。

## 6. 観測可能範囲

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 6.1 | 一定距離以内のみ観測（斜距離基準） | ✅ | `doppler_observation`：3次元斜距離 ≤ R_max で探知 | `test_out_of_range_is_explicit_non_detection` |
| 6.2 | 上限斜距離は変数、全観測者で共通 | ✅ | `max_slant_range_yd`（GIS / API） | `test_health_and_configuration_round_trip` |
| 6.3 | 範囲内は欠測なし | ✅ | 範囲内は毎秒必ず観測。範囲外は明示的な非探知を送信し、推定器はこれを情報として使用 | — |

## 7. 音源・ドップラー・最近接

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 7.1 | 音源周波数は一定 | ✅ | `source_frequency_hz` | — |
| 7.2 | ドップラー計測自体の誤差なし | ✅ | 観測周波数は厳密値（推定器は数値分解能 0.03 Hz を尤度幅に使用） | `test_doppler_frequency_sign` |
| 7.3 | 取得間隔1秒、全観測者で同期 | ✅ | doppler コンテナは目標と全観測者が同一 tick に揃うのを待って一括生成 | — |
| 7.4 | 音源演算部は別コンテナ | ✅ | `doppler` サービス | — |
| 7.5 | 最近接時に最近接斜距離を得る（ドップラーで判断） | ✅ | `CpaAnalyzer`：認識周波数との零交差＝最近接時刻、傾き＋前後変化＝斜距離 | `test_cpa_exact_without_bias` |
| 7.6 | 最近接前後のドップラー変化量から相対速力 | ✅ | `classical_cpa`：f(tc−τ) − f(tc+τ) と傾きから V, R | 同上 |
| 7.7 | 最近接斜距離の誤差要因：音源周波数誤差・速力誤差 | ✅ | `bias_range_sigma_yd`（f′±σ_b で再計算）、`speed_range_sigma_yd`（2R·σ_V/V） | `test_cpa_range_speed_and_bias_shift` |
| 7.8 | 最近接時刻の誤差要因：真の周波数と認識周波数の差 | ✅ | `bias_shift_tick_s` ≈ b / |df/dt| | 同上（シフト量を検証） |
| 7.9 | 周波数認識誤差：全観測者共通・時間変化なし・大きさ未設定 | 🔧 | 真値 `shared_recognition_bias_hz`（既定 0）、推定側は状態 b として推定（事前 σ 既定 0.5 Hz） | `test_common_frequency_bias_is_estimated` |

## 8. 算出結果・過去航跡

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 8.1 | 現在位置・過去航跡・速度・深度・不確かさ | ✅ | `TrackEstimate`（位置、深度±σ、HDG/COG±σ、速度±σ、水平1σ長短径、track[]） | 推定器テスト |
| 8.2 | 過去航跡を更新する／しないの2通り | ✅ | `SMOOTHED`（祖先追跡による固定ラグ平滑化）／`ONLINE`（凍結） | `test_online_track_frozen_and_smoothed_track_updated` |
| 8.3 | 過去航跡の再計算範囲は可変 | ✅ | `smoothing_window_seconds`（GIS / API） | 同上（範囲外は凍結を検証） |
| 8.4 | 観測者との相対速度表示 | ✅ | `relative[]`（観測者ごと）＋ CPA の相対速力 | GIS 表 |
| 8.5 | 対地速度・対水速度表示 | ✅ | `ground_speed_kt` / `through_water_speed_kt` | GIS |
| 8.6 | 目標の HDG と COG（CUS→COG） | ✅ | `hdg_deg`（対水速度の向き）/ `cog_deg`（対地速度の向き） | GIS |
| 8.7 | ドップラー（＋方位）観測から位置・深度・速力・航跡・存在圏 | ✅ | `TrackingEngine` 全体 | 推定器テスト |

## 9. 不確かさ・存在圏・単位

| # | 要件 | 状態 | 実装 |
|---|---|---|---|
| 9.1 | 存在確率を％で設定 | ✅ | `presence_probability_pct` |
| 9.2 | **最終指定：楕円（体）ではなく推定存在圏を表示** | ✅ | `estimation/region.py`：最高密度領域（HDR）。分離領域も個別に確率表示 |
| 9.3 | 描画形式（未指定、点群指定なし） | 🔧 | 各連結成分の**水平外形を深度範囲で押し出した立体**＋HDRボクセル（表示切替可） |
| 9.4 | 距離 YD、深度 Ft | ✅ | API・GIS とも YD / Ft |
| 9.5 | 速度表示単位（独立指定なし） | 🔧 | **kt**（速力変化率と同じ）。深度変化率は Ft/s |

## 10. システム構成・設計・実装

| # | 要件 | 状態 | 実装 |
|---|---|---|---|
| 10.1 | Docker コンテナ環境 | ✅ | `docker-compose.yml`（db, api, clock, current-field, target, observer×N, doppler, estimator, web） |
| 10.2 | GIS（オープンソース）、Cesium | ✅ | CesiumJS（Apache-2.0）＋ Natural Earth II（パブリックドメイン、オフライン同梱） |
| 10.3 | 合成データのシミュレーター | ✅ | 真値生成と推定を分離（推定器には真値を渡さない：`test_estimator_feed_contains_no_truth`）。真値は比較表示のみに使用 |
| 10.4 | 目標と観測者は独立コンテナ | ✅ | `target`、`observer`（1観測者1コンテナ） |
| 10.5 | 音源演算部は必要に応じ別コンテナ | ✅ | `doppler` |
| 10.6 | UML（PlantUML） | ✅ | `docs/uml/*.puml`（コンポーネント・クラス・シーケンス・推定アクティビティ・観測者状態） |
| 10.7 | リポジトリ UeEmon/aqua-drift-simulator | ✅ | 本リポジトリ |

## 11. 比較表示・推定操作

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 11.1 | **推定状態と真の目標情報を並べて比較** | ✅ | GIS「推定と真値」タブ：位置・深度・HDG・COG・対水/対地速力・深度変化率・周波数偏りを「推定／真値／誤差／推定1σ」で並列表示、真値が存在圏の内外どちらか、誤差と1σの時間推移グラフ、観測者ごとの相対速力・斜距離・方位（推定／真値）、最近接（推定／真値）、流速場（推定／真値）。地図に推定–真値の誤差線 | UI スモークテスト |
| 11.2 | **推定の開始・停止を操作** | ✅ | `POST /api/estimation/start`（現在時刻以降の観測で新しい推定ランを開始）／`/stop`（停止、最後の結果を保持表示）。GIS 上部に開始・停止ボタンと状態表示。起動時の自動開始は `ESTIMATION_AUTOSTART` | `test_estimation_start_stop_runs`, `test_estimation_control_endpoints` |

## 12. 描画・視点・表示

| # | 要件 | 状態 | 実装 | 検証 |
|---|---|---|---|---|
| 12.1 | **GPU による描画改善** | ✅ | 毎秒のエンティティ再生成をやめ、PolylineCollection / PointPrimitiveCollection / LabelCollection（バッチ描画）へ移行。存在圏の外形・ボクセルは一括ジオメトリ（Primitive）として Web Worker で生成し、完成後に差し替え（ちらつきなし）。必要時のみ描画（requestRenderMode）、MSAA 4x＋FXAA、端末解像度での描画、高性能GPU優先のWebGLコンテキスト、FPS表示 | CI ヘッドレス E2E（WebGL2・MSAA・描画モードを確認） |
| 12.2 | **視点切り替え（水平方向・垂直方向）** | ✅ | 地図右上のツールバー：斜視／真上（垂直）／水平（側面、方向は「進行方向の真横」または東西南北から選択）、真上は正射投影を選択可 | E2E：真上でピッチ −90°、水平で 0° を確認 |
| 12.3 | **目標推定位置を画面中心にするボタン** | ✅ | 「◎ 推定位置を中心」（現在の向き・距離を保って移動）、「真値を中心」、「推定位置に追従」 | E2E：ボタン後に推定位置の画面座標が中心から 5 % 以内 |
| 12.4 | **推定と真値の比較を横スクロールなしで一覧** | ✅ | 2 列のカード表示（誤差を大きく表示し、推定・真値・1σ を併記、|誤差|/1σ で「1σ内／3σ内／3σ超」判定）。表は折り返し表示で横スクロールなし。パネル幅はドラッグで変更可 | E2E：1600 px / 1280 px で横方向のはみ出しがないことを確認 |

## 未確定事項（ユーザー判断待ち）

| 項目 | 現在の扱い |
|---|---|
| 針路変化率の単位 | 度/秒 |
| 流速場の時間更新規則 | 時間一定（設定変更で更新） |
| 周波数認識誤差の大きさ | 既定 0 Hz（GIS で設定可）。推定側事前 σ 0.5 Hz |
| 存在圏の描画形式 | 外形立体＋ボクセル |
| 推定開始時に過去の観測を使うか | 使わない（開始時刻以降の観測のみ） |
| 周囲4点の半径・向き | 3000 YD、目標針路に対して斜め45°（設定 `deployment.surround_radius_yd`） |
| 位置・距離観測の誤差（比較用） | 位置 50 YD / 30 Ft、距離 2 %（比較ツールの仮定値） |
| シミュレーター名称 | リポジトリ名に合わせ「AQUA-DRIFT」 |
