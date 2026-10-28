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
| 4.27（続き） | 変針・ジグザグの失探の対策。推定器の移動の処理に旋回の運動モデル（等速→一定の旋回率→等速の仮説を当てはめ、等速と尤度を混合。粒子の状態は 7 要素のまま。`move_turn_mode` 既定 `after_maneuver`：機動の検出から 300 秒）。推定は変針に追いつくようになったが（500 YD・S1 で変針 2 分後の誤差 4〜16 YD）、それだけでは探知の途切れは変わらず、設標者が変針前の旧針路の線を置き続けていたのが原因だった。機動の検出後は、設標者が向かっている・間近の投下でも目標が近づかなくなったものを取り消す（`replan_obsolete_after_maneuver`、自動了承のみ、`DeploymentRequest.obsolete`）。さらに設標者が旋回待機を 1 周して投下が約 250 秒遅れる 2 つの原因を直した（投下直後に真正面の次の投下点へそのまま向かう `layer.passes_ahead`、流される投下点で飛行中の経路を保つ条件を捕捉半径の 0.8 倍に `KEEP_PATH_FRACTION`）。再探知投下は設標者の経路に沿って順に計画し、着く時刻を予定時刻にした（`search_deployment.search_points_routed`）。500 YD・各 4 シードの探知なしの時間：90° 変針 46→21 %、ジグザグ 63→38 %、直進・変速・深度変化は 6〜11 %（ばらつきの範囲）。6,000 YD は途切れ 0 %、誤差は同等か小さい | `estimation/move.py`（`_turn_back`、`find_modes`）、`estimation/particle_filter.py`（`_turn_modes`）、`replanning.py`（`obsolete_drops`）、`layer.py`、`search_deployment.py`、`state.py`、`services/deployer.py` |
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

* **変針・ジグザグの失探（4.27、大きく改善・残課題あり）**：推定器の旋回モデル、機動後の旧針路の投下の取り消し、設標者の
  旋回待機 1 周の 2 つの原因の修正、再探知投下の経路化（以上 PR #50）、探知が戻った後の再探知投下の取り消し、線の投下時刻の
  見積もりで向かっている区間を飛んでいるとおりに扱う修正（`feature/redetection-after-loss`）で、500 YD の探知なしの時間は
  90° 変針 46→20 %（8 シード）、ジグザグ 63→41 %（8 シード、19〜66 %）、直進 6 %・変速 4 %・深度変化 4 % になった。
  探知なしの時間のほとんどは LOST の前の COASTING で、全実行に開始直後の約 90 秒（3.75 %）の途切れがある（設標者の初期位置
  からの飛行時間）。残る主因は、探知の切れ目に機動が重なると推定できずに失探すること（S4 seed 3 は版ごとに 27 % か 65 %）と、
  単独の投下（線でない 1〜2 本）の後に設標者が 1 周して次の線が遅れること（S0 seed 0、1610 秒から 220 秒）。
  機動のシナリオは 4 シードでは楽観的に出ることがある（S1 で 22 % と 28 %）ので 8 シードで比べる。
  単独の投下が目標の後ろに置かれる原因（線の順路の 2 区間目以降を正確な経路で見積もり、実際より 1 周遅く出る）を直す
  `loose_all` を試したが、直進は良くなった一方で機動のシナリオが悪化したため取り消した（`docs/optimal-deployment.md`
  の「試して取り消したもの」）。次の候補は、線の投下時刻を目標が来る直前にすること。
  B 案（粒子の状態に旋回率を加える）は、推定がすでに変針に追いついているため効果は小さいと見ている。
  旋回の当てはめの開始点を減らす軽量化を試したが、計算時間の差は測定のばらつきの範囲で S4 が 43→50 % とばらつきの範囲で
  悪く出たため取り消した。6,000 YD・S4 seed 0 を 3 版同時に流した実時間は旋回モデルの有無で同じ（約 1,740 秒）。
  なお、4 並列の閉ループの実時間（`wall_s`）は numpy の多重スレッドの奪い合いで大きく揺れる（負荷平均 5〜7）ので、
  別々に流した実行の実時間は比べられない。比べるときは同時に流すか `OMP_NUM_THREADS=1` にする（4 並列でも速くなる）。
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
