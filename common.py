#!/usr/bin/env python3
"""common.py - generate.py / generate_gui.py / dist_tensors.py / lora_chance_ui.py で共有する純粋ユーティリティ。

このモジュールはスクリプトエントリーポイントを持たず、argparse / main() ループも含まない。
ComfyUI HTTP 移行 (2026-05〜06) で旧 diffusers パイプライン関連 (sd_playground 期) は撤去済み。
ComfyUI playground 側に残った機能だけを保持している。

提供するもの:
    定数:   PROMPT_TOML / LORA_PARAM_TOML
    TOML:   load_prompt_config / load_lora_params
    プロンプト: normalize_emphasis / build_prompt / EMPHASIS_RE 系
    LoRA 抽選: build_lora_corpus / pick_lora_by_keywords / pick_n_loras_by_keywords
    解析:   classify_tensor / detect_base_arch / detect_vae_arch / lora_target_arch /
            detect_controlnet_arch / detect_embedding_arch / extract_lora_trigger_hints
            (系統は "flux1" / "flux2" / "sdxl" / "sd15" / "unknown" を返す)
    変換:   convert_to_safetensors / file_sha256
    dtype:  pick_device_dtype (Ampere+ で bf16、それ未満で fp16、CPU で fp32)
    監視:   current_gpu_temp
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import subprocess
from pathlib import Path
from typing import Optional

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib

import torch
from PIL import Image
from safetensors.torch import save_file

# コンソール出力の言語切替 (英/日)。再エクスポートして from common import L を可能にする。
from i18n import L, LANG, set_lang  # noqa: F401

# --------------------------------------------------------------------------- #
# 定数 / ディレクトリ構成
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent
PROMPT_TOML = ROOT / "prompt.toml"
LORA_PARAM_TOML = ROOT / "LoRA_param.toml"

# `*word*` / `**word**` / `***word***` を捉える。先頭側の `*` 数で重みが決まる。
EMPHASIS_RE = re.compile(r"(\*+)([^*]+)\*+")
# asterisk 数 → compel 重み (markdown の italics/bold/bold-italics に倣う段階表現)
_EMPHASIS_WEIGHTS = {1: 1.1, 2: 1.3}  # それ以上 (3+) は 1.5 にフォールバック


# fp16 で読んだ場合のおおよその常駐サイズ (GB)。アクティベーション余裕 1.5GB を別途見込む
# version 別のディテール優先デフォルト解像度


# --------------------------------------------------------------------------- #
# プロンプト
# --------------------------------------------------------------------------- #
def normalize_emphasis(text: str) -> str:
    """`*word*` / `**word**` / `***word***` を compel の重み記法 `(word:1.x)` に変換する。
    asterisks の数で強度: 1=1.1, 2=1.3, 3+=1.5。
    compel が prompt_embeds 構築時にこの記法をパースして per-token weight に変換する。
    """
    def _repl(m: re.Match) -> str:
        n = len(m.group(1))
        weight = _EMPHASIS_WEIGHTS.get(n, 1.5)
        return f"({m.group(2)}:{weight})"
    return EMPHASIS_RE.sub(_repl, text)


def load_prompt_config() -> dict:
    """prompt.toml をロード。存在しなければ最小デフォルトを返す。"""
    if not PROMPT_TOML.exists():
        return {
            "who": [["a girl", False]],
            "wearing": [["nothing", 1]],
            "with_items": [["nothing", 1]],
            "motion": [["standing", 1]],
            "at": ["room"],
            "lighting": ["bright lighting"],
            "positive_always": "",
            "negative_always": "",
        }
    with open(PROMPT_TOML, "rb") as f:
        return tomllib.load(f)


def _wpick_entry(pairs: list) -> list | None:
    """[[value, weight, kw?], ...] から 1 つ重み抽選してエントリ全体を返す。
    pairs 空なら None。各エントリは可変長 ([value, weight] or [value, weight, kw])。
    """
    if not pairs:
        return None
    weights = [max(0, int(p[1])) for p in pairs]
    if sum(weights) == 0:
        return random.choice(pairs)
    return random.choices(pairs, weights=weights, k=1)[0]


def _entry_value(entry: list | None) -> str:
    return str(entry[0]) if entry else ""


def _entry_keyword(entry: list | None) -> str:
    """エントリの 3 番目要素 (LoRA キーワード) を返す。無ければ空文字。"""
    if entry and len(entry) >= 3 and entry[2]:
        return str(entry[2])
    return ""


def build_prompt(cfg: dict) -> tuple[str, str, list[str], bool]:
    """prompt.toml の設定から (positive, negative, lora_keywords, many) を組み立てる。

    many は採用された who エントリが複数人 (例 "2 women") を表すフラグ。
    生成側はこれを見て横長キャンバスに切り替える等の判断に使う。

    各セクションの動作:
      who         : [character, has_wearing, has_motion?, has_where?, many?, kw?] から均等ランダム 1 つ。
                    has_wearing=true なら wearing セクション自体をスキップ。
                    has_motion=true なら motion をスキップ。
                    has_where=true なら at セクションをスキップ (= キャラ文字列に場所が内包、例
                    "a girl ware swimsuit at pool" / "a girl ware swimsuit in sea")。
                    many=true は複数人 (キャラ文字列が "2 women" 等)。生成側で横長化に使う。
                    後方互換: 3〜5 要素目の **型** で版を判別:
                      - [char, has_motion, kw_str]            旧 v01 形式 (3 要素、kw のみ。bool 1 個は has_motion 扱い)
                      - [char, has_wearing, has_motion, kw_str]    v02 形式 (4 要素、has_where=false default)
                      - [char, has_wearing, has_motion, has_where, kw_str]  v03 形式 (5 要素、index4=str)
                      - [char, has_wearing, has_motion, has_where, many, kw_str]  v04 形式 (6 要素、index4=bool)
      wearing     : [clothing, weight, kw?] から重み抽選 1 つ。"nothing" → "naked"、他は "wearing X"。
                    who.has_wearing=true ならこのセクションは走らない。
      with_items  : [item, weight, kw?] から 3 回独立重み抽選 → dedupe / "nothing" 除外 → 各 "with X"。
      motion      : [motion, weight, kw?] から重み抽選 1 つ (has_motion=false のときのみ)。
      at          : 均等抽選 1 つ → "at {place}" (has_where=false のときのみ)。
      lighting    : 均等抽選 1 つ → "with {lighting}"。
      *_always    : 末尾 / negative 本体。

    最後の kw 要素は当該エントリが採用されたときに LoRA 名マッチング用キーワードとして
    収集される (`pick_lora_by_keywords` で使用)。最終的に normalize_emphasis で
    `**word**` を compel 重み記法に変換して返す。
    """
    parts: list[str] = []
    lora_keywords: list[str] = []

    # 誰が: [char, has_wearing, has_motion?, has_where?, kw?]
    # 後方互換性のため、3〜5 要素を型判定で振り分ける。
    # v01 (3 要素 [char, has_motion, kw]) は単独 bool を has_motion として扱う旧仕様。
    # v02/v03 (4-5 要素) では [char, has_wearing, has_motion, ...] の順に変更済み。
    who_entries = cfg.get("who") or []
    has_motion = False
    has_wearing = False
    has_where = False
    many = False
    if who_entries:
        # v05 形式: [char, weight(int), has_wearing, has_motion, has_where, many, kw_str]
        # 旧形式 (v04/v03/v02/v01) は weight 列無し → 一律 weight=1 で重み抽選
        def _who_weight(e):
            if len(e) >= 2 and isinstance(e[1], (int, float)) and not isinstance(e[1], bool):
                return max(1, int(e[1]))
            return 1
        chosen = random.choices(who_entries,
                                weights=[_who_weight(e) for e in who_entries], k=1)[0]
        char = str(chosen[0]) if chosen else ""
        kw_value: str | None = None
        is_v05 = (len(chosen) >= 2 and isinstance(chosen[1], (int, float))
                  and not isinstance(chosen[1], bool))
        if is_v05:
            has_wearing = bool(chosen[2]) if len(chosen) >= 3 else False
            has_motion  = bool(chosen[3]) if len(chosen) >= 4 else False
            has_where   = bool(chosen[4]) if len(chosen) >= 5 else False
            many        = bool(chosen[5]) if len(chosen) >= 6 else False
            if len(chosen) >= 7 and chosen[6]:
                kw_value = str(chosen[6])
        else:
            # 旧形式パース (3 要素目で v01 vs v02+ を判定。bool=v02+ / str=v01)
            if len(chosen) >= 3:
                third = chosen[2]
                if isinstance(third, bool):
                    # v02+: [char, has_wearing, has_motion, has_where?, many?, kw?]
                    has_wearing = bool(chosen[1]) if len(chosen) >= 2 else False
                    has_motion = third
                    if len(chosen) >= 4:
                        fourth = chosen[3]
                        if isinstance(fourth, bool):
                            has_where = fourth
                            if len(chosen) >= 5:
                                fifth = chosen[4]
                                if isinstance(fifth, bool):
                                    many = fifth
                                    if len(chosen) >= 6 and chosen[5]:
                                        kw_value = str(chosen[5])
                                elif fifth:
                                    kw_value = str(fifth)
                        else:
                            if fourth:
                                kw_value = str(fourth)
                else:
                    # v01: [char, has_motion, kw]
                    has_motion = bool(chosen[1]) if len(chosen) >= 2 else False
                    if third:
                        kw_value = str(third)
            elif len(chosen) >= 2:
                has_motion = bool(chosen[1])
        if char:
            parts.append(char)
        if kw_value:
            lora_keywords.append(kw_value)

    # 何を着て: has_wearing=true ならセクション全体をスキップ (= キャラ文字列に服装情報が
    # 内包されている前提、wearing 抽選自体を回さない)。
    # has_wearing=false は通常抽選: "nothing" → "naked"、その他は "wearing {clothing}"。
    if not has_wearing:
        w_entry = _wpick_entry(cfg.get("wearing") or [])
        w = _entry_value(w_entry)
        if w == "nothing":
            parts.append("naked")
        elif w:
            parts.append(f"wearing {w}")
        kw = _entry_keyword(w_entry)
        if kw:
            lora_keywords.append(kw)

    # with アクセサリ/状況 (最大 3、dedupe、"nothing" 除外)
    with_pool = cfg.get("with_items") or []
    if with_pool:
        weights = [max(0, int(p[1])) for p in with_pool]
        if sum(weights) > 0:
            entries = random.choices(with_pool, weights=weights, k=3)
        else:
            entries = [random.choice(with_pool) for _ in range(3)]
        seen: set[str] = set()
        for ent in entries:
            it = str(ent[0])
            if not it or it == "nothing" or it in seen:
                continue
            seen.add(it)
            parts.append(f"with {it}")
            if len(ent) >= 3 and ent[2]:
                lora_keywords.append(str(ent[2]))

    # 動作 (キャラ側に動作が無いときのみ)
    if not has_motion:
        m_entry = _wpick_entry(cfg.get("motion") or [])
        m = _entry_value(m_entry)
        if m:
            parts.append(m)
        kw = _entry_keyword(m_entry)
        if kw:
            lora_keywords.append(kw)

    # at 場所 (has_where=true なら キャラ文字列に既に場所が含まれているのでスキップ)
    # 各エントリは "place"(str, weight=1 default) または [place, weight, kw_str]
    if not has_where:
        at_list = cfg.get("at") or []
        if at_list:
            at_pool: list[str] = []
            at_w: list[int] = []
            at_kw: list[str] = []
            for e in at_list:
                if isinstance(e, str):
                    at_pool.append(e); at_w.append(1); at_kw.append("")
                elif isinstance(e, (list, tuple)) and e:
                    at_pool.append(str(e[0]))
                    at_w.append(max(1, int(e[1])) if len(e) >= 2 else 1)
                    at_kw.append(str(e[2]) if len(e) >= 3 else "")
            if at_pool:
                idx = random.choices(range(len(at_pool)), weights=at_w, k=1)[0]
                parts.append(f"at {at_pool[idx]}")
                if at_kw[idx]:
                    lora_keywords.append(at_kw[idx])

    # with 明るさ
    lighting_list = cfg.get("lighting") or []
    if lighting_list:
        parts.append(f"with {random.choice(lighting_list)}")

    # 表情 / ムード (均等抽選 1 つ、そのまま 1 句として付加)。毎枚バリエーションを出す。
    expression_list = cfg.get("expression") or []
    if expression_list:
        parts.append(str(random.choice(expression_list)))

    # 必ず付加 (positive)
    pos_always = str(cfg.get("positive_always") or "").strip()
    if pos_always:
        parts.append(pos_always)

    neg_always = str(cfg.get("negative_always") or "").strip()

    # LoRA キーワード重複排除: 各 entry をカンマ区切りで atomic に分解、
    # 大小文字無視で初出順を保つ。複数 entry に同じ kw (例: "nude") が混じっても 1 回のみ。
    seen_kws: set[str] = set()
    deduped_kws: list[str] = []
    for kw_entry in lora_keywords:
        for atom in str(kw_entry).split(","):
            atom = atom.strip()
            if not atom:
                continue
            key = atom.lower()
            if key in seen_kws:
                continue
            seen_kws.add(key)
            deduped_kws.append(atom)

    return (
        normalize_emphasis(", ".join(parts)),
        normalize_emphasis(neg_always),
        deduped_kws,
        many,
    )


# --------------------------------------------------------------------------- #
# LoRA_param.toml
# --------------------------------------------------------------------------- #
def load_lora_params() -> dict[str, dict]:
    """LoRA_param.toml をロードして {LoRA stem: {"trigger": "...", ...}} を返す。"""
    if not LORA_PARAM_TOML.exists():
        return {}
    with open(LORA_PARAM_TOML, "rb") as f:
        return tomllib.load(f) or {}


def build_lora_corpus(loras: list[Path], lora_params: dict) -> dict[str, str]:
    """各 LoRA の検索用 lowercase 文字列を `{stem: corpus}` で返す。`pick_lora_by_keywords` が参照。

    マッチ対象は 3 つの情報源を結合したテキスト:
      1. **LoRA のファイル名** (stem)
      2. **プリセットキーワード**: `extract_lora_trigger_hints` で safetensors メタから抽出した
         activation_text / trigger_words / ss_tag_frequency 上位タグ
      3. **LoRA_param.toml の trigger**: ユーザが設定した手動 trigger 文字列

    起動時に 1 回作って LoRA 数ぶん回しても 200 件で 1〜2 秒程度 (メタ読みのみ)。
    """
    corpus: dict[str, str] = {}
    for lora in loras:
        stem = lora.stem
        try:
            hints = extract_lora_trigger_hints(lora)
        except Exception:
            hints = []
        trigger = str((lora_params.get(stem) or {}).get("trigger") or "")
        parts = [stem, *hints, trigger]
        corpus[stem] = " ".join(p for p in parts if p).lower()
    return corpus


def _parse_keyword_clauses(keywords: list[str]) -> list[list[str]]:
    """keyword 文字列群を AND/OR の入れ子に展開する:
      - `,` で分割 → OR の clause リスト
      - 各 clause を空白で分割 → AND の token リスト
      - 全 lowercase 化、空文字は除外

    例: ['naked, nude', 'swimsuit beach', 'wet']
        → [['naked'], ['nude'], ['swimsuit', 'beach'], ['wet']]
           (naked OR nude OR (swimsuit AND beach) OR wet の意)
    """
    clauses: list[list[str]] = []
    for kw in keywords:
        for clause_str in str(kw).split(","):
            tokens = [t.strip().lower() for t in clause_str.split() if t.strip()]
            if tokens:
                clauses.append(tokens)
    return clauses


def _text_matches_clauses(text: str, clauses: list[list[str]]) -> bool:
    """text を OR of ANDs で判定 (各 clause 内は AND、clause 間は OR)。
    text は事前に lowercase 化済み前提 (本関数では追加変換しない)。
    """
    for clause in clauses:
        if all(token in text for token in clause):
            return True
    return False


def pick_lora_by_keywords(compat_loras: list[Path], keywords: list[str],
                          corpus: dict[str, str] | None = None) -> Optional[Path]:
    """compat_loras から keywords にマッチする LoRA を抽選する。

    **keyword 構文** (大小文字を問わない):
      - `,` 区切り = **OR** (どれかの clause がマッチすれば該当)
      - 空白区切り = **AND** (clause 内の全 token が含まれる必要あり)
      - 例: `"naked oiled, bondage"` → (naked AND oiled) OR bondage

    - 各 LoRA について `corpus[stem]` (build_lora_corpus 製、stem + preset hints + trigger) を
      lowercase で部分一致検査 (corpus 未指定なら stem 単独で検査、後方互換)。
    - match があれば **97% で match 候補から random.choice**、3% で全 compat_loras から random.choice
      (= 従来抽選にフォールバック、用語に縛られない多様性確保)。
    - match が無ければ 100% 全 compat_loras から random.choice (= 従来抽選と同義)。
    - 呼び出し側で `args.lora_prob` ゲートを通す前提。compat_loras が空なら None。
    """
    if not compat_loras:
        return None
    clauses = _parse_keyword_clauses(keywords)
    if clauses:
        matches = []
        for lora in compat_loras:
            text = (corpus.get(lora.stem) if corpus is not None else None) or lora.stem.lower()
            if _text_matches_clauses(text, clauses):
                matches.append(lora)
        if matches and random.random() < 0.97:
            return random.choice(matches)
    return random.choice(compat_loras)


def pick_n_loras_by_keywords(
    compat_loras: list[Path],
    keywords: list[str],
    corpus: dict[str, str] | None = None,
    n_max: int = 3,
    n_min: int = 1,
) -> list[Path]:
    """n_min〜n_max 個のユニークな LoRA を重ね掛け用に抽選する (n は random.randint で決まる)。

    **1 pick = 1 キーワード** の原則。キーワード列をシャッフルして先頭から消費し、各 pick で
    `pick_lora_by_keywords` を **その 1 キーワードだけ** で呼ぶ。これによって 1 回の生成内で
    同じキーワードが 2 回以上 LoRA 抽選に使われない (大小文字無視で dedup 済み)。

    - keywords が空 → 全 LoRA から random.choice で n_target 個
    - keywords が n_target 未満 → 全 kw を使い切り、実 n はその数に縮む
    - 各 pick で `pick_lora_by_keywords` の 97% match / 3% random フォールバックは生きる
    - 既に選ばれた LoRA は次の pick の候補から除外、同じ LoRA は重複しない

    呼び出し側は: ① n=1 なら従来通り 1 LoRA を fuse、② n>1 なら set_adapters で複数を adapter
    として読み、scales = [args.lora_scale / n] * n で重ね掛けする想定。
    """
    if not compat_loras:
        return []
    # n_min/n_max を [1, len(compat_loras)] にクランプ + n_min <= n_max を保証
    hi = max(1, min(n_max, len(compat_loras)))
    lo = max(1, min(n_min, hi))
    n_target = random.randint(lo, hi)

    # キーワードを大小無視で dedup + シャッフル (build_prompt 側で既に dedup 済みでも安全側に再実行)
    seen: set[str] = set()
    unique_kws: list[str] = []
    for kw in (keywords or []):
        atom = str(kw).strip()
        if not atom:
            continue
        key = atom.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_kws.append(atom)
    random.shuffle(unique_kws)

    picked: list[Path] = []
    picked_stems: set[str] = set()

    if not unique_kws:
        # キーワード無し: 純ランダム pick を n_target 回
        for _ in range(n_target):
            candidates = [l for l in compat_loras if l.stem not in picked_stems]
            if not candidates:
                break
            chosen = random.choice(candidates)
            picked.append(chosen)
            picked_stems.add(chosen.stem)
        return picked

    # 1 pick = 1 kw 消費、kw が尽きたら停止 (実 n は min(n_target, len(unique_kws)))
    n_actual = min(n_target, len(unique_kws))
    for i in range(n_actual):
        candidates = [l for l in compat_loras if l.stem not in picked_stems]
        if not candidates:
            break
        chosen = pick_lora_by_keywords(candidates, [unique_kws[i]], corpus)
        if chosen is None:
            break
        picked.append(chosen)
        picked_stems.add(chosen.stem)
    return picked


def extract_lora_trigger_hints(lora_path: Path) -> list[str]:
    """LoRA safetensors のメタデータから候補 trigger を最大 5 個抽出する。
    取得できなければ空リスト。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(lora_path), framework="pt") as f:
            meta = f.metadata() or {}
    except Exception:
        return []
    candidates: list[str] = []
    # 直接的なフィールド (フォーマットによっては存在)
    for key in ("activation_text", "trigger_words", "ss_keep_tokens"):
        v = meta.get(key)
        if v:
            candidates.append(str(v))
    # ss_tag_frequency: {subset: {tag: count}} を JSON で持つ
    tf_raw = meta.get("ss_tag_frequency")
    if tf_raw:
        try:
            tf_data = json.loads(tf_raw)
            counts: dict[str, int] = {}
            for subset in (tf_data or {}).values():
                if isinstance(subset, dict):
                    for tag, c in subset.items():
                        try:
                            counts[tag] = counts.get(tag, 0) + int(c)
                        except (TypeError, ValueError):
                            continue
            top = sorted(counts.items(), key=lambda x: -x[1])[:5]
            candidates.extend(t for t, _ in top)
        except Exception:
            pass
    # 重複排除しつつ順序保持。数字のみ / "None" / 過度に短い等のノイズは弾く。
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        c = c.strip().strip('"').strip("'").strip()
        if not c or c in seen:
            continue
        if c.lower() == "none":
            continue
        if c.isdigit():
            continue
        if len(c) < 3:  # '1', 'a' 等の単独文字
            continue
        if len(c) > 60:
            continue
        seen.add(c)
        out.append(c)
    return out[:5]


# --------------------------------------------------------------------------- #
# safetensors の分類・チェック
# --------------------------------------------------------------------------- #
# --- Flux 判定ヘルパー -------------------------------------------------------- #
# Flux の DiT (transformer) ブロック名。区切り非依存の部分一致で拾う:
#   BFL/Comfy : "...double_blocks.0..."  /  "...single_blocks.0..."
#   kohya LoRA: "lora_unet_double_blocks_0_..."  (アンダースコア区切り)
#   diffusers : "...transformer_blocks..."  /  "...single_transformer_blocks..."
_FLUX_DIT_MARKERS = ("double_blocks", "single_blocks", "transformer_blocks")
# flux.1 dev/schnell の隠れ次元 (img_in / x_embedder の出力)。F2 構造判定の基準。
_FLUX1_HIDDEN = 3072
# text encoder hidden dims (Embedding 判別用)。4096 = T5-XXL (Flux)。
_EMB_DIMS = {768, 1024, 1280, 2048, 4096}


def _strip_prefix(key: str) -> str:
    """ラッパ prefix を剥がして素のキー名にする。"""
    for pre in ("model.diffusion_model.", "model.", "first_stage_model.", "vae.", "module."):
        if key.startswith(pre):
            return key[len(pre):]
    return key


def _arch_fields(meta: dict | None) -> str:
    """系統判定に効く厳選フィールドのみを小文字連結。全値ブロブは使わない
    (URL 末尾の "...flux" と次値 "2" が "flux 2" に化ける誤検知を防ぐため)。"""
    if not meta:
        return ""
    keys = ("modelspec.architecture", "ss_base_model_version",
            "modelspec.title", "ss_network_module")
    return " ".join(str(meta.get(k, "")) for k in keys).lower()


def _is_flux_dit(keys: list[str]) -> bool:
    return any(any(m in k for m in _FLUX_DIT_MARKERS) for k in keys)


def _flux_hidden(f, keys: list[str]) -> int | None:
    """img_in / x_embedder の出力次元 (= 隠れ次元) を返す。取れなければ None。"""
    for k in keys:
        if _strip_prefix(k).endswith(("img_in.weight", "x_embedder.weight")):
            shape = list(f.get_slice(k).get_shape())
            if shape:
                return int(shape[0])
    return None


def _flux_signal(f, keys: list[str], meta: dict | None) -> str | None:
    """Flux 系なら世代 ("flux1" / "flux2") を、Flux でなければ None を返す。

    判定順:
      1. ss_base_model_version=="flux2" / architecture に flux2・flux-2・flux.2 → flux2
      2. 厳選フィールドに "flux" or キーに flux DiT ブロック名 → flux 系と確定
      3. flux 系で隠れ次元が 3072 以外 → flux2 (暫定ヒューリスティック)
      4. それ以外の flux 系 → flux1 / flux でなければ None
    NOTE: Flux.2 の実テンソルがまだ手元に無いため 3. は暫定。実物入手後にキー署名へ置換すること。
    """
    af = _arch_fields(meta)
    bmv = str((meta or {}).get("ss_base_model_version", "")).lower()
    if bmv.startswith("flux2") or any(t in af for t in ("flux2", "flux-2", "flux.2")):
        return "flux2"
    if not (("flux" in af) or _is_flux_dit(keys)):
        return None
    h = _flux_hidden(f, keys)
    if h is not None and h != _FLUX1_HIDDEN:
        return "flux2"
    return "flux1"


def _is_vae(keys: list[str]) -> bool:
    """VAE (Flux ae / AutoencoderKL) か。encoder.* / decoder.* のみで構成され DiT/UNet を含まない。
    all-in-one checkpoint は VAE を内包するが double_blocks 等も併せ持つので all() で除外される。"""
    if not keys:
        return False
    cores = [_strip_prefix(k) for k in keys]
    if not any(c.startswith(("encoder.", "decoder.")) for c in cores):
        return False
    extra = {"quant_conv.weight", "quant_conv.bias",
             "post_quant_conv.weight", "post_quant_conv.bias"}
    return all(c.startswith(("encoder.", "decoder.")) or c in extra for c in cores)


# --------------------------------------------------------------------------- #
def _gguf_arch(path: Path) -> str:
    """GGUF (Flux unet 等) の系統を返す: "flux1" / "flux2" / "unknown"。
    可能なら gguf メタ (general.architecture) を読み、無ければファイル名で推定。"""
    n = path.stem.lower()
    if any(t in n for t in ("flux2", "flux-2", "flux.2")):
        return "flux2"
    try:
        from gguf import GGUFReader  # ComfyUI-GGUF と同梱の gguf pkg
        r = GGUFReader(str(path))
        fld = r.fields.get("general.architecture")
        if fld is not None:
            arch = str(bytes(fld.parts[fld.data[0]]), "utf-8", "replace").lower()
            if "flux" in arch:
                return "flux1"
    except Exception:
        pass
    return "flux1" if "flux" in n else "unknown"


def classify_tensor(path: Path) -> str:
    """base / lora / embedding / controlnet / vae / inpainting / broken を返す。
    safetensors のヘッダ (キー + shape) のみ参照しメモリには展開しない。
    GGUF は unet (Flux transformer) 前提で base 扱い。
    """
    if path.suffix.lower() == ".gguf":
        return "base" if _gguf_arch(path) != "unknown" else "broken"
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            if not keys:
                return "broken"
            # ControlNet 判別 (LoRA より先): diffusers (controlnet_*) / A1111 (control_model.* /
            # zero_convs) / LLLite (lllite_*) / Flux ControlNet (controlnet_blocks /
            # controlnet_x_embedder / input_hint_block) を拾う。
            if any(k.startswith("controlnet_") or "controlnet_cond_embedding" in k
                   or "control_model." in k or "zero_convs" in k
                   or "controlnet_blocks" in k or "controlnet_x_embedder" in k
                   or "input_hint_block" in k
                   or k.startswith("lllite_") for k in keys):
                return "controlnet"
            if any("lora_" in k or ".lora_" in k for k in keys):
                return "lora"
            if any(k in {"emb_params", "string_to_param"} or k.startswith("string_to_param") for k in keys):
                return "embedding"
            # Textual Inversion / Embedding: キー数が極端に少ない (base は数百〜数千キー)。
            # 各テンソルは 1D、または 2D なら最終次元が text encoder の hidden dim。
            # 例: ng_deepnegative_v1_75t は [75,768] (prod 57600 で旧 50k 閾値を超えるが TI embedding)。
            if len(keys) <= 8:
                shapes = [list(f.get_slice(k).get_shape()) for k in keys]

                def _is_emb_shape(s: list[int]) -> bool:
                    if len(s) == 1:
                        return True
                    if len(s) == 2:
                        return s[-1] in _EMB_DIMS or math.prod(s) < 50_000
                    return False

                if shapes and all(_is_emb_shape(s) for s in shapes):
                    return "embedding"
            # VAE (Flux ae / AutoencoderKL)。all-in-one checkpoint はここを通らず base へ。
            if _is_vae(keys):
                return "vae"
        # inpainting checkpoint (UNet 入力 9ch、通常 pipe では使えない) を名前で除外
        if "inpaint" in path.stem.lower():
            return "inpainting"
        return "base"
    except Exception:
        return "broken"


def detect_base_arch(path: Path) -> str:
    """checkpoint の系統を返す: "flux1" / "flux2" / "sdxl" / "sd15" / "unknown"。ヘッダのみ参照。"""
    if path.suffix.lower() == ".gguf":
        return _gguf_arch(path)
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata()
            sig = _flux_signal(f, keys, meta)
            if sig:
                return sig
            # 旧 SD 系 (lowtensors 行き)
            if any("conditioner.embedders" in k or "text_encoder_2" in k for k in keys):
                return "sdxl"
            for k in keys:
                if k.endswith(("to_k.weight", "to_q.weight")):
                    shape = list(f.get_slice(k).get_shape())
                    if shape and shape[-1] >= 2048:
                        return "sdxl"
            if any("input_blocks" in k or "down_blocks" in k or "diffusion_model" in k for k in keys):
                return "sd15"
    except Exception:
        return "unknown"
    return "unknown"


def detect_vae_arch(path: Path) -> str:
    """VAE の系統を返す: "flux" (16ch latent) / "sd" (4ch) / "unknown"。

    encoder.conv_out.weight の出力 ch = 2 * latent_channels (mean+logvar)。
    Flux ae: 16ch → 32 / SD VAE: 4ch → 8。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if _strip_prefix(k).endswith("encoder.conv_out.weight"):
                    shape = list(f.get_slice(k).get_shape())
                    if shape:
                        out = int(shape[0])
                        if out >= 24:
                            return "flux"
                        if out <= 12:
                            return "sd"
    except Exception:
        return "unknown"
    return "unknown"


def detect_controlnet_arch(path: Path) -> str:
    """ControlNet の系統を返す: "flux1" / "flux2" / "sdxl" / "sd15" / "unknown"。

    Flux ControlNet は DiT ブロック (double_blocks 等) や controlnet 専用キーを持つ。
    SD 系は cross-attention context 次元 (to_k/to_q/to_v 最終次元) で判定: 2048=SDXL / 768=SD15。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata()
            sig = _flux_signal(f, keys, meta)
            if sig or any("controlnet_blocks" in k or "controlnet_x_embedder" in k for k in keys):
                return sig or "flux1"
            if any("text_encoder_2" in k or "_te2_" in k for k in keys):
                return "sdxl"
            for k in keys:
                if k.endswith(("to_k.weight", "to_q.weight", "to_v.weight")):
                    shape = list(f.get_slice(k).get_shape())
                    if shape:
                        inner = shape[-1]
                        if inner >= 1500:
                            return "sdxl"
                        if 600 <= inner <= 800:
                            return "sd15"
    except Exception:
        return "unknown"
    return "unknown"


def lora_target_arch(path: Path) -> str:
    """LoRA の対象系統を返す: "flux1" / "flux2" / "sdxl" / "sd15" / "unknown"。

    Flux LoRA は metadata (ss_base_model_version=flux1 / networks.lora_flux 等) か、
    double_blocks / single_blocks を対象にしたキー (lora_unet_double_blocks_* /
    diffusion_model.*.lora_down 等) で判る。te1 のみ学習した Flux LoRA もあるので metadata を優先。
    SD 系は cross-attn context 次元 (lora_down 最終次元) で 2048=SDXL / 768=SD15。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            meta = f.metadata()
            sig = _flux_signal(f, keys, meta)
            if sig:
                return sig
            if any("text_encoder_2" in k or "_te2_" in k or "lora_te2_" in k
                   or "clip_g" in k.lower() for k in keys):
                return "sdxl"
            dims: set[int] = set()
            for k in keys:
                if ("to_k" in k or "to_v" in k) and ("lora_down.weight" in k or "lora_A.weight" in k):
                    shape = list(f.get_slice(k).get_shape())
                    if len(shape) >= 2:
                        dims.add(int(shape[-1]))
            if any(d >= 1500 for d in dims):   # cross-attn context 2048 → SDXL
                return "sdxl"
            if 768 in dims:                     # cross-attn context 768 → SD15
                return "sd15"
    except Exception:
        return "unknown"
    return "unknown"


def detect_embedding_arch(path: Path) -> str:
    """Embedding (Textual Inversion) の系統を返す: "flux1" / "sdxl" / "sd15" / "unknown"。

    CLIP-L (768) ベースは Flux でも利用可 → flux1 扱い (3_5 行き)。T5 (4096) も Flux → flux1。
    SDXL は bigG (1280) を含む二段構造 → sdxl 扱い (lowtensors 行き)。
    """
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = list(f.keys())
            if any("clip_g" in k.lower() or "_te2_" in k.lower() or "text_encoder_2" in k for k in keys):
                return "sdxl"
            dims: set[int] = set()
            for k in keys:
                shape = list(f.get_slice(k).get_shape())
                if shape:
                    dims.add(int(shape[-1]))
            if 1280 in dims:
                return "sdxl"
            if 4096 in dims or 768 in dims:
                return "flux1"
    except Exception:
        return "unknown"
    return "unknown"


def _flatten_tensors(obj, prefix: str = "") -> dict[str, torch.Tensor]:
    """ネストした dict から Tensor だけを再帰的に拾い、ドット区切りキーで平坦化"""
    out: dict[str, torch.Tensor] = {}
    if isinstance(obj, torch.Tensor):
        out[prefix or "tensor"] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_tensors(v, key))
    return out


def convert_to_safetensors(path: Path) -> Path:
    """.ckpt / .pt を .safetensors に変換。a1111 TI 形式や生 state_dict にも対応。"""
    out = path.with_suffix(".safetensors")
    raw = torch.load(str(path), map_location="cpu", weights_only=False)

    if isinstance(raw, torch.Tensor):
        sd = {path.stem: raw}
    elif isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        sd = {k: v for k, v in raw["state_dict"].items() if isinstance(v, torch.Tensor)}
    elif isinstance(raw, dict) and "string_to_param" in raw:
        # a1111 Textual Inversion 形式: string_to_param['*'] に Tensor
        params = raw.get("string_to_param") or {}
        token = raw.get("name") or path.stem
        tensor = params.get("*") if isinstance(params, dict) else None
        if isinstance(tensor, torch.Tensor):
            sd = {str(token): tensor}
        else:
            sd = _flatten_tensors(params, prefix=str(token))
    elif isinstance(raw, dict):
        # フラットな state_dict
        sd = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
        if not sd:
            sd = _flatten_tensors(raw)
    else:
        raise ValueError(L(f"対応できない形式: {type(raw).__name__}", f"unsupported format: {type(raw).__name__}"))

    if not sd:
        raise ValueError(L("Tensor が見つかりません", "no tensors found"))
    sd = {k: v.contiguous() for k, v in sd.items()}
    save_file(sd, str(out))
    return out


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def pick_device_dtype(vram_gb: float) -> tuple[str, torch.dtype]:
    """SD パイプラインの weight dtype を決める。
    Ampere (sm_80) 以降は bf16、それ未満の CUDA は fp16、CPU は fp32。

    bf16 は fp32 と同じ指数幅 (8bit) のため、SD15 系で fp16 UNet が overflow して
    灰色画像 (NaN latent → grey) を吐く問題が出ない。Ampere ではスループットも
    fp16 と同等なので、対応 GPU では bf16 を既定とする。
    vram_gb は呼び出し側の build_pipeline 等で slicing/offload 判定に使う。
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return "cuda", torch.bfloat16
        return "cuda", torch.float16
    return "cpu", torch.float32


# --------------------------------------------------------------------------- #
# 出力 / 画像判定
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# GPU 温度
# --------------------------------------------------------------------------- #
def current_gpu_temp() -> Optional[int]:
    """nvidia-smi で取れる GPU 温度 (°C)。取得失敗時は None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None
