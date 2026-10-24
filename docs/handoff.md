# 開発の引き継ぎ（AQUA-DRIFT）

新しいセッションで開発を続けるための要点です。最終更新：2026-10-24（ブランチ `feature/redetection-search` 時点）。

## 1. リポジトリと進め方

| 項目 | 内容 |
|---|---|
| リポジトリ | `UeEmon/aqua-drift-simulator` |
| 作業ブランチ | 直近は `feature/redetection-search`（main から分岐。main へ直接 push しない）。以前は `feature/doppler-estimation` |
| PR | 直近は `feature/redetection-search` → main の PR（番号は第 7 節）。main にはこれまでの作業がマージ済み |
| CI | `.github/workflows/ci.yml`：backend（ruff + pytest）、compose、web、e2e（Playwright、最大 60 分）、soak（25 分の耐久） |
| 最新の CI | `edda74c` で 5 ジョブすべて成功 |
| 応答 | ユーザーへの報告・UI 文言は日本語。単位は YD / Ft / kt / 度 |
| コミット | 末尾に Co-Authored-By と Claude-Session の行（セッションの指示に従う） |

## 2. 作業環境の準備と確認

```bash
cd backend
pip install --break-system-packages -e ".[dev]"   # PyPI に接続できること
python -m pytest -q                               # 全テスト（数分）
ruff check backend                                # CI と同じ lint（リポジトリ直下で）
node --check web/app.js
```

* 前のセッションでは途中から組織の通信ポリシーにより pypi.org / files.pythonhosted.org /
  registry.npmjs.org が 403 になり、手元で FastAPI 依存のテストを実行できなかった
  （CI で全件確認）。403 は迂回せず、ユーザーに報告すること。
* Docker での全体起動：`docker compose up --build`（17 コンテナ、http://localhost:8080）。
* E2E：`e2e/ui_check.py`（画面の段階チェック）、`e2e/soak_check.py`（耐久）、`e2e/reuse_check.py`。

## 3. システムの概要（詳細は docs/）

* ドップラーのみによる水中目標の 3 次元追尾シミュレーター。漂流する観測者（ソノブイ相当）の
  ドップラー観測から、正則化粒子フィルタ（7 状態：位置・対水速度・周波数偏り）で推定。
* Web 画面（CesiumJS、`web/`）：WebSocket 差分プロトコル v2（`web/stream-decoder.js`）。
  品質の自動調整、追従カメラと HUD、なめらか表示。
* 主要ドキュメント：`docs/requirements.md`（要求の追跡表。新機能は行を追加し、実装・検証を記入）、
  `docs/architecture.md`、`docs/optimal-deployment.md`、`docs/depth-from-doppler.md`、
  `docs/estimation-methods.md`、`docs/uml/`。

## 4. 直近に実装した機能（新しい順）

| 要求 | 内容 | 主なファイル |
|---|---|---|
| 4.27 | 変針・変速・深度変化に対応した配置計画と設標者の飛行の見直し（短い探知距離）。直線敷設は、設標者が空くまで `line_queue_ahead_s`（60 秒）を超える間は次の線を計画しない、線の評価期間を設標者の 1 周期から決める（`line_max_per_pass` 8 本）、到着時刻を誘導と同じ経路で見積もる（`LayerAvailability.earliest_s`：ゆるい経路・捕捉半径、向きを問わない点は 36 方位の最短）、設標者が航跡上を同じ向きに飛ぶときはその飛行線の延長に置く。機動の直後と COASTING の間は機動の仮説を今から始め、航跡を横切る線も候補にし、前程計画・再計画を続ける。推定器の機動応答の設定 `move_maneuver_cut_fraction`・`maneuver_respread_fraction` を追加（既定は従来どおり）。500 YD の探知なしの時間：直進 22→8 %、変速 15→6 %、深度変化 35→11 %、90° 変針 55→46 %、ジグザグ 76→63 %。6,000 YD は回帰なし。詳細は `docs/optimal-deployment.md` | `optimal_deployment.py`、`forward_deployment.py`、`replanning.py`、`estimation/particle_filter.py`、`analysis/deployment_diagnosis.py` |
| 4.26 | 失探知（LOST）の間の再探知投下：コーストした粒子雲を探知円で貪欲に覆う（`search_deployment.py`、`replanning.plan_search_replacement`、`DropTask.source = "search"`、画面の表示「再探知」）。残す前程投下の後に予定する。閉ループでは探知を戻せていない（投下は真値から 1,800〜8,000 YD、4.27 の原因による） | `search_deployment.py`、`replanning.py`、`services/deployer.py`、`scenario.py` |
| UI | 右パネルのタブを作業ごとに再編：推定結果／目標／観測・推定／配置・設標（前程配置の状況と設定・設標計画・設標者・手動配置）／風・潮流（潮流の設定・平均風の推定・風向風速）／表示。設定フォームはタブごとに分け、各タブの「反映」はそのタブの項目だけを送る（`config-form`・`forward-form`・`current-form`） | `web/index.html`、`web/app.js`、`web/styles.css` |
| 4.18 | 風向風速（海面〜30,000 ft、1,000 ft ごと）。設標者は高度を持って飛び（巡航・投下高度、昇降率）飛行高度の風で流される。観測者は投下後に各高度の風を受けて自由落下し、着水時刻に投入。着水位置と無風の自由落下の予想地点の差から投下高度〜海面の平均風を推定し、次の投下点を風上に修正。「風・潮流」タブ。詳細は `docs/wind.md` | `wind.py`、`layer.py`（`release_point`、気団座標の誘導）、`state.py`（`falling`、`_splash`）、`web/app.js`（`updateWind`） |
| 4.16 | 設標順の入れ替え：「配置・設標」タブの計画一覧で ▲▼。投入時刻は新しい順で再計算（最適時刻 `requested_tick` か前の設標点から到着できる時刻の遅い方）。API `/api/drops/reorder` | `state.py`（`reorder_tasks`）、`models.py`（`DropTask.flight_key`）、`web/app.js`（`moveDrop`） |
| 4.15 | 設標者の旋回は左右どちらでもよく、最大バンク角の範囲で経路の短い側を選ぶ（基準旋回 `preferred_turn`・`turn_margin_s` は廃止、古い設定に残っていても無視）。待機旋回の向きは旋回に入るとき少ない旋回で乗れる側（`LayerState.orbit_sense`）。「配置・設標」タブ（状態表、有無・一時停止・了承方式・速力などの設定、計画一覧で了承／却下／今すぐ／時刻変更／中止、手動配置）。API `/api/drops/cancel`・`/api/drops/reschedule` | `backend/aqua_drift/layer.py`（`dubins_turn`、`step` の待機旋回）、`state.py`、`api.py`、`web/index.html`（`tab-deploy`）、`web/app.js`（`updateLayer`） |
| 4.14 | 最適な投入予定時刻を計算し、設標者は最短経路の到着が計画時刻に最も近い時に旋回待機から出発し、経路・速力を変えずに到着時に設標 | `optimal_deployment.py`（`LayerAvailability`、`drop_times`）、`layer.advance` |
| 4.13 | 追加の観測者は設標者（基準速力 200 kt 一定、バンク 15° 以内）が配置。提案→了承（既定は自動）→設標。待機中は推定位置の周囲を旋回 | `layer.py`、`services/layer.py`、`models.py`（`LayerConfig`、`DropTask`） |
| 4.12 | 観測者の最適自動投入（位置・本数・深度をフィッシャー情報量と探知距離から計算） | `optimal_deployment.py`、`forward_deployment.py`、`services/deployer.py` |
| — | ロイドミラー（直接波と海面反射波の干渉）による深度推定。処理が重いため設定でオン／オフ | `estimation/lloyd.py`、`physics.py`、`engine.py` |
| 12.x | カメラ HUD、追従（既定オン・真値）、初期高度 10,000 ft、地図の縦領域を画面内に、GPU 過負荷による固まり対策 | `web/app.js`、`web/styles.css` |

## 5. 未完了・注意事項

* **リポジトリの非公開化**：プロキシが設定変更を拒否したため未実施。ユーザーに手順
  （GitHub の Settings → General → Danger Zone → Change visibility）を案内済み。迂回しないこと。
* `ruff check e2e` には既存の指摘が残っている（CI の対象は `backend` のみ）。
* 設標者の計画時刻との差：時間は飛行経路（左旋回の遠回り）で合わせ、速力は経路で合わないときだけ
  変える方式にした。間に合う計画の 99%（177/179）が ±5 秒以内、最大 12 秒。速力を変えるのは
  投下点が旋回円の内側に入るなどの 34% の区間（主に減速、中央値約 30 kt）。
* UI のスモークテスト：`e2e/ui_smoke.py`（CI の `ui-smoke` ジョブ）。バックエンドなしで web/ を配信し、
  ヘッドレス Chromium で Cesium の 3D 画面が起動して地球を描画することを確かめる（事前に `web/` で `npm ci`）。
* フィクスチャ生成：`backend/aqua_drift/analysis/wire_fixture.py`。

* **変針・ジグザグで探知が途切れる（未解決、4.27）**：500 YD の直線敷設では探知中の観測者が 1〜2 本・一直線で、
  推定器の移動の処理（`estimation/move.py`、移動窓 600 秒の等速の当てはめ）が変針に 1〜2 分遅れ、しかも自信過剰
  （変針 60 秒後に推定 90°・1σ 1.4°、真値 151°）。配置計画側の対策（機動の直後・COASTING の計画、横切る線）は
  効果がばらつきの範囲だった。次の候補は推定器に旋回の運動モデルを入れること：A 案は `move.py` の当てはめを
  「等速＋一定の旋回率」にする（見積もり 5〜9 時間）、B 案は粒子の状態に旋回率を加える（9〜13 時間）。
  **ユーザーは今回の修正の効果を見てから判断する**としている（2026-10-24）。
* **効果を確かめる道具**：`python -m aqua_drift.analysis.deployment_diagnosis --scenarios S0,S1,S2,S3,S4 --range 500
  --seeds 4 --seconds 2400 --particles 1000 --jobs 4 --out 出力.jsonl`（`--set` で前程配置、`--est` で推定器の設定を上書き。
  投下ごとの計画・着水・真値との距離・探知の有無と、探知なしの時間の割合を出す）。4 並列で 500 YD は 20 実行約 40 分、
  6,000 YD は約 45 分。2 シードではばらつきが大きく判断を誤ったので、比較は 4 シード以上で行う。
  `analysis/lost_contact.py` には `nosearch` と `fwd:key=value` の変種がある。
* **作業環境**：クラウドのコンテナは途中で再起動することがあり、実行中のプロセスが止まる。長い検証は
  `--out` に結果を逐次保存し、同じコマンドで続きから再開できる。依存は第 2 節の手順で入れ直す
  （再起動後も入っていたが、無ければ `pip install -e '.[dev]'`）。全テストは約 30〜38 分。
* **GitHub 連携**：このセッションでは GitHub のプラグインがプロキシで接続できなかった（403）。`git push` と `gh` は使えた。

## 6. 新しいセッションの最初の手順

1. リポジトリを取得し、作業するブランチを確認する（`feature/redetection-search` の PR がマージ済みなら main から新しいブランチを切る）。
2. この文書と `docs/requirements.md` の 4.12〜4.16、4.26〜4.27、`docs/optimal-deployment.md` の「通過単位の計画と待ち行列」以降を読む。
3. 第 2 節の手順で依存を入れ、全テストが通ることを確認（PyPI が 403 ならユーザーに報告）。
4. `gh run list --branch <作業ブランチ> --limit 1` で最新の CI が成功していることを確認。
5. ユーザーの次の指示を待つ。

## 7. 直近の PR

* `feature/redetection-search` → main：[UeEmon/aqua-drift-simulator#49](https://github.com/UeEmon/aqua-drift-simulator/pull/49)。要求 4.26（再探知投下）と 4.27（配置計画と設標者の飛行の見直し）。
