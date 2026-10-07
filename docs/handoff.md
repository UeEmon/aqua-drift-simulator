# 開発の引き継ぎ（AQUA-DRIFT）

新しいセッションで開発を続けるための要点です。最終更新：2026-10-08（コミット `edda74c` 時点）。

## 1. リポジトリと進め方

| 項目 | 内容 |
|---|---|
| リポジトリ | `UeEmon/aqua-drift-simulator` |
| 作業ブランチ | `feature/doppler-estimation`（main へ直接 push しない） |
| PR | #1（すべての作業をこの PR に積む） |
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
| 4.15 | 設標者の旋回は左旋回が基準。右旋回は経路が `turn_margin_s`（既定 10 秒）× 速力以上短い場合だけ。「設標者」タブ（状態表、有無・一時停止・了承方式・基準旋回・速力などの設定、計画一覧で了承／却下／今すぐ／時刻変更／中止、手動配置）。API `/api/drops/cancel`・`/api/drops/reschedule` | `backend/aqua_drift/layer.py`（`dubins_turn`、`preferred_side`）、`state.py`、`api.py`、`web/index.html`（`tab-layer`）、`web/app.js`（`updateLayer`） |
| 4.14 | 最適な投入予定時刻を計算し、設標者が計画時刻に設標（出発待ち、速力選択、HOLD） | `optimal_deployment.py`（`LayerAvailability`、`drop_times`）、`layer.advance` |
| 4.13 | 追加の観測者は設標者（200±50 kt、バンク 15° 以内）が配置。提案→了承（既定は自動）→設標。待機中は推定位置の周囲を旋回 | `layer.py`、`services/layer.py`、`models.py`（`LayerConfig`、`DropTask`） |
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

## 6. 新しいセッションの最初の手順

1. リポジトリを取得し、`feature/doppler-estimation` をチェックアウト。
2. この文書と `docs/requirements.md` の 4.12〜4.15 を読む。
3. 第 2 節の手順で依存を入れ、全テストが通ることを確認（PyPI が 403 ならユーザーに報告）。
4. `gh run list --branch feature/doppler-estimation --limit 1` で最新の CI が成功していることを確認。
5. ユーザーの次の指示を待つ。
