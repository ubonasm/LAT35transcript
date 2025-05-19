import streamlit as st
import pandas as pd
import spacy
import re
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import altair as alt

# 必要なライブラリをインポートセクションに追加
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import io
import base64
from PIL import Image
import tempfile
import streamlit.components.v1 as components

# アプリケーションのタイトル
st.set_page_config(page_title="授業記録分析ツール", layout="wide")
st.title("授業記録分析ツール")

# セッション状態の初期化
if 'data' not in st.session_state:
    st.session_state.data = None
if 'concepts' not in st.session_state:
    st.session_state.concepts = {}
if 'nlp' not in st.session_state:
    st.session_state.nlp = None
if 'language' not in st.session_state:
    st.session_state.language = "ja"
if 'model_size' not in st.session_state:
    st.session_state.model_size = "sm"

# 言語モデルの基本名
language_base = {
    "ja": "ja_core_news",
    "en": "en_core_web",
    "zh": "zh_core_web",
    "ko": "ko_core_news",
    "fr": "fr_core_news"
}

# モデルサイズの辞書
model_sizes = {
    "sm": "小 (sm)",
    "md": "中 (md)",
    "lg": "大 (lg)"
}

# サイドバー - 言語選択
language_options = {
    "ja": "日本語",
    "en": "英語",
    "zh": "中国語",
    "ko": "韓国語",
    "fr": "フランス語"
}
selected_language = st.sidebar.selectbox(
    "言語を選択してください",
    options=list(language_options.keys()),
    format_func=lambda x: language_options[x],
    index=0
)

# サイドバー - モデルサイズ選択
selected_size = st.sidebar.selectbox(
    "モデルサイズを選択してください",
    options=list(model_sizes.keys()),
    format_func=lambda x: model_sizes[x],
    index=0,
    help="大きいモデルほど精度が高くなりますが、読み込みと処理に時間がかかります"
)

# 言語またはサイズが変更されたときにモデルを再ロードする条件を更新します:
if selected_language != st.session_state.language or selected_size != st.session_state.model_size:
    st.session_state.language = selected_language
    st.session_state.model_size = selected_size
    st.session_state.nlp = None


# 言語モデルのロード
@st.cache_resource
def load_nlp_model(lang, size):
    model_name = f"{language_base[lang]}_{size}"
    try:
        return spacy.load(model_name)
    except OSError:
        st.warning(f"{language_options[lang]}の{model_sizes[size]}モデルをダウンロードしています...")
        spacy.cli.download(model_name)
        return spacy.load(model_name)


if st.session_state.nlp is None:
    st.session_state.nlp = load_nlp_model(selected_language, selected_size)

nlp = st.session_state.nlp

# 現在のモデル情報を表示
st.sidebar.info(f"現在のモデル: {language_base[selected_language]}_{selected_size}")

# アプリケーションの説明
st.markdown("""
## 授業記録分析ツール

このツールは、CSV形式の授業記録を分析します。
**CSVファイル要件:**
- 「発言番号」「発言者」「発言内容」の3つの列が必要です
- 文字コードはUTF-8を推奨します

**分析機能:**
1. **KWIC分析**: 特定の単語やフレーズの前後の文脈を表示
2. **概念辞書作成**: ANDとORを使った概念の定義
3. **概念分析・可視化**: 概念に基づく発言の分析と可視化
4. **ネットワーク分析**: 概念と発言の関係性をネットワーク図で表示
""")

# ファイルアップロード（より目立つように装飾）
st.sidebar.markdown("## CSVファイルのアップロード")
st.sidebar.markdown("授業記録のCSVファイルをアップロードしてください。")

uploaded_file = st.sidebar.file_uploader("", type=["csv"], help="「発言番号」「発言者」「発言内容」の列が必要です",
                                         key="csv_uploader")

if uploaded_file is not None:
    try:
        # エンコーディングオプションを提供
        encoding_options = ["utf-8", "shift-jis", "cp932", "euc-jp"]
        selected_encoding = st.sidebar.selectbox(
            "ファイルのエンコーディング",
            options=encoding_options,
            index=0
        )

        # 区切り文字オプションを提供
        delimiter_options = [",", "\t", ";"]
        selected_delimiter = st.sidebar.selectbox(
            "区切り文字",
            options=delimiter_options,
            index=0,
            format_func=lambda x: "カンマ (,)" if x == "," else "タブ (\\t)" if x == "\t" else "セミコロン (;)"
        )

        # CSVファイルを読み込み
        data = pd.read_csv(uploaded_file, encoding=selected_encoding, delimiter=selected_delimiter)

        # 必要なカラムがあるか確認
        required_columns = ["発言番号", "発言者", "発言内容"]
        if not all(col in data.columns for col in required_columns):
            # カラム名のマッピングを提供
            st.sidebar.warning("必要なカラムが見つかりません。カラムをマッピングしてください。")

            column_mapping = {}
            for req_col in required_columns:
                column_mapping[req_col] = st.sidebar.selectbox(
                    f"{req_col}に対応するカラムを選択",
                    options=data.columns,
                    key=f"map_{req_col}"
                )

            # カラム名を変更
            data = data.rename(columns=column_mapping)

        # データを保存
        st.session_state.data = data
        st.sidebar.success(f"ファイルが正常に読み込まれました。発言数: {len(data)}件")

        # 基本統計情報を表示
        st.sidebar.markdown("### 基本統計")
        speaker_counts = data["発言者"].value_counts()
        st.sidebar.markdown(f"- 総発言数: {len(data)}件")
        st.sidebar.markdown(f"- 発言者数: {len(speaker_counts)}人")

    except Exception as e:
        st.sidebar.error(f"ファイルの読み込み中にエラーが発生しました: {e}")
        st.sidebar.info("別のエンコーディングや区切り文字を試してみてください。")

# 概念辞書のエクスポート/インポート機能
st.sidebar.markdown("---")
st.sidebar.subheader("概念辞書のエクスポート/インポート")

# エクスポート
if st.session_state.concepts and st.sidebar.button("概念辞書をエクスポート"):
    concepts_json = json.dumps(st.session_state.concepts, ensure_ascii=False, indent=2)
    st.sidebar.download_button(
        label="JSONファイルをダウンロード",
        data=concepts_json,
        file_name="concept_dictionary.json",
        mime="application/json"
    )

# インポート
uploaded_concepts = st.sidebar.file_uploader("概念辞書をインポート", type=["json"], key="concept_uploader")
if uploaded_concepts is not None:
    try:
        imported_concepts = json.load(uploaded_concepts)
        st.session_state.concepts.update(imported_concepts)
        st.sidebar.success("概念辞書がインポートされました。")
    except Exception as e:
        st.sidebar.error(f"インポート中にエラーが発生しました: {e}")

# CSVファイルのアップロード状態を表示
if st.session_state.data is not None:
    st.success(f"CSVファイルが読み込まれています。発言数: {len(st.session_state.data)}件")

    # データサンプルを表示
    with st.expander("データプレビュー"):
        st.dataframe(st.session_state.data.head())
else:
    st.info("サイドバーからCSVファイルをアップロードしてください。")

# タブの作成
tab1, tab2, tab3, tab4 = st.tabs(["KWIC分析", "概念辞書作成", "概念分析・可視化", "ネットワーク分析"])

# KWIC分析タブ
with tab1:
    st.header("KWIC分析")

    if st.session_state.data is not None:
        search_term = st.text_input("検索語を入力してください")

        if search_term:
            results = []

            for idx, row in st.session_state.data.iterrows():
                text = str(row["発言内容"])
                if search_term in text:
                    doc = st.session_state.nlp(text)
                    tokens = [token.text for token in doc]

                    for i, token in enumerate(tokens):
                        if search_term in token:
                            start = max(0, i - 20)
                            end = min(len(tokens), i + 21)

                            context_before = " ".join(tokens[start:i])
                            context_after = " ".join(tokens[i + 1:end])

                            results.append({
                                "発言番号": row["発言番号"],
                                "発言者": row["発言者"],
                                "前文脈": context_before,
                                "検索語": token,
                                "後文脈": context_after
                            })

            if results:
                df_results = pd.DataFrame(results)

                # 結果の表示
                for _, row in df_results.iterrows():
                    st.markdown(f"**発言番号**: {row['発言番号']} - **発言者**: {row['発言者']}")
                    st.markdown(f"{row['前文脈']} <span style='color:red'>{row['検索語']}</span> {row['後文脈']}",
                                unsafe_allow_html=True)
                    st.markdown("---")
            else:
                st.info(f"「{search_term}」は見つかりませんでした。")
    else:
        st.info("CSVファイルをアップロードしてください。")

# 概念辞書作成タブ
with tab2:
    st.header("概念辞書作成")

    if st.session_state.data is not None:
        st.markdown("""
        概念辞書を作成します。例: 「救急」という概念は「(病気 OR 痛み OR 風邪) AND (急ぎ OR 急病 OR 救急車)」のように表現できます。
        """)

        concept_name = st.text_input("概念名を入力してください")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("AND条件グループ")
            and_groups = []

            # 既存の概念を編集する場合
            editing_concept = None
            if concept_name and concept_name in st.session_state.concepts:
                editing_concept = st.session_state.concepts[concept_name]
                and_groups = editing_concept.get("and_groups", [])

            # 動的にANDグループを追加
            num_and_groups = st.number_input("ANDグループ数", min_value=1, value=len(and_groups) if and_groups else 1)

            new_and_groups = []
            for i in range(int(num_and_groups)):
                st.markdown(f"**ANDグループ {i + 1}**")

                # 既存のOR条件を取得
                existing_or_terms = and_groups[i] if i < len(and_groups) else []

                # OR条件の入力
                or_terms = st.text_area(
                    f"OR条件（各行に1つの単語）#{i}",
                    value="\n".join(existing_or_terms) if existing_or_terms else "",
                    height=100,
                    key=f"or_group_{i}"
                )

                # 空白行を除去して配列に変換
                or_terms_list = [term.strip() for term in or_terms.split("\n") if term.strip()]
                new_and_groups.append(or_terms_list)

        # 概念の保存
        if st.button("概念を保存"):
            if concept_name:
                st.session_state.concepts[concept_name] = {
                    "name": concept_name,
                    "and_groups": new_and_groups
                }
                st.success(f"概念「{concept_name}」が保存されました。")
            else:
                st.error("概念名を入力してください。")

        # 概念の削除
        if st.button("概念を削除") and concept_name in st.session_state.concepts:
            del st.session_state.concepts[concept_name]
            st.success(f"概念「{concept_name}」が削除されました。")

        with col2:
            st.subheader("保存された概念")
            if st.session_state.concepts:
                for name, concept in st.session_state.concepts.items():
                    with st.expander(name):
                        expression = []
                        for i, or_group in enumerate(concept["and_groups"]):
                            or_expr = " OR ".join([f'"{term}"' for term in or_group])
                            expression.append(f"({or_expr})")

                        st.markdown(" AND ".join(expression))
            else:
                st.info("保存された概念はありません。")

# 概念分析・可視化タブ
with tab3:
    st.header("概念分析・可視化")

    if st.session_state.data is not None and st.session_state.concepts:
        # 可視化方法の選択
        viz_method = st.radio(
            "可視化方法を選択してください",
            ["概念ごとの発言一覧", "発言ごとの概念マトリックス"]
        )

        # 概念に基づく発言の分析
        concept_matches = {}

        for concept_name, concept in st.session_state.concepts.items():
            matches = []

            for idx, row in st.session_state.data.iterrows():
                text = str(row["発言内容"]).lower()

                # 各ANDグループについて確認
                all_and_conditions_met = True

                for or_group in concept["and_groups"]:
                    any_or_condition_met = False

                    for term in or_group:
                        if term.lower() in text:
                            any_or_condition_met = True
                            break

                    if not any_or_condition_met:
                        all_and_conditions_met = False
                        break

                if all_and_conditions_met:
                    matches.append({
                        "発言番号": row["発言番号"],
                        "発言者": row["発言者"],
                        "発言内容": row["発言内容"]
                    })

            concept_matches[concept_name] = matches

        # 可視化
        if viz_method == "概念ごとの発言一覧":
            for concept_name, matches in concept_matches.items():
                with st.expander(f"概念: {concept_name} ({len(matches)}件)"):
                    if matches:
                        df_matches = pd.DataFrame(matches)
                        st.dataframe(df_matches)
                    else:
                        st.info(f"概念「{concept_name}」に該当する発言はありません。")

        else:  # 発言ごとの概念マトリックス
            # マトリックスデータの作成
            matrix_data = []

            for idx, row in st.session_state.data.iterrows():
                utterance_data = {
                    "発言番号": row["発言番号"],
                    "発言者": row["発言者"],
                    "発言内容": row["発言内容"]
                }

                for concept_name in st.session_state.concepts.keys():
                    # 該当する発言かどうかを確認
                    matches = concept_matches[concept_name]
                    utterance_data[concept_name] = "○" if any(m["発言番号"] == row["発言番号"] for m in matches) else ""

                matrix_data.append(utterance_data)

            # データフレームに変換して表示
            df_matrix = pd.DataFrame(matrix_data)
            st.dataframe(df_matrix)

            # ヒートマップの作成
            st.subheader("概念出現ヒートマップ")

            # ヒートマップ用のデータ準備
            concept_names = list(st.session_state.concepts.keys())
            utterance_numbers = df_matrix["発言番号"].tolist()

            heatmap_data = []
            for idx, row in df_matrix.iterrows():
                for concept in concept_names:
                    value = 1 if row[concept] == "○" else 0
                    heatmap_data.append({
                        "発言番号": row["発言番号"],
                        "概念": concept,
                        "該当": value
                    })

            df_heatmap = pd.DataFrame(heatmap_data)

            # Altairでヒートマップを作成
            chart = alt.Chart(df_heatmap).mark_rect().encode(
                x=alt.X('概念:N', title='概念'),
                y=alt.Y('発言番号:O', title='発言番号', sort='ascending'),
                color=alt.Color('該当:Q', scale=alt.Scale(domain=[0, 1], range=['white', 'red'])),
                tooltip=['発言番号', '概念', '該当']
            ).properties(
                width=600,
                height=min(1000, len(utterance_numbers) * 15)
            )

            st.altair_chart(chart, use_container_width=True)

    elif not st.session_state.data:
        st.info("CSVファイルをアップロードしてください。")
    elif not st.session_state.concepts:
        st.info("概念辞書を作成してください。")

# ネットワーク分析タブ
with tab4:
    st.header("ネットワーク分析")

    if st.session_state.data is not None and st.session_state.concepts:
        # ネットワーク可視化の種類を選択
        network_type = st.radio(
            "ネットワーク分析の種類を選択してください",
            ["概念用語ネットワーク", "概念-発言ネットワーク", "発言者-概念ネットワーク", "多次元尺度法(MDS)"]
        )

        # 概念用語ネットワーク
        if network_type == "概念用語ネットワーク":
            st.subheader("概念用語の共起ネットワーク")
            st.write("概念辞書内の用語間の関連性を可視化します。同じ概念内で使用される用語は関連が強いとみなされます。")

            # ネットワークの設定
            min_edge_weight = st.slider("最小エッジの重み", 1, 10, 1,
                                        help="この値以上の共起回数を持つエッジのみを表示します")

            # 用語の共起関係を計算
            terms = set()
            term_to_concepts = defaultdict(list)

            # すべての用語を収集
            for concept_name, concept in st.session_state.concepts.items():
                for or_group in concept["and_groups"]:
                    for term in or_group:
                        terms.add(term)
                        term_to_concepts[term].append(concept_name)

            # 共起行列を作成
            term_list = list(terms)
            cooccurrence_matrix = np.zeros((len(term_list), len(term_list)))

            for concept_name, concept in st.session_state.concepts.items():
                concept_terms = set()
                for or_group in concept["and_groups"]:
                    concept_terms.update(or_group)

                # 同じ概念内の用語間の共起をカウント
                concept_term_list = list(concept_terms)
                for i in range(len(concept_term_list)):
                    for j in range(i + 1, len(concept_term_list)):
                        term1 = concept_term_list[i]
                        term2 = concept_term_list[j]
                        idx1 = term_list.index(term1)
                        idx2 = term_list.index(term2)
                        cooccurrence_matrix[idx1, idx2] += 1
                        cooccurrence_matrix[idx2, idx1] += 1

            # NetworkXグラフを作成
            G = nx.Graph()

            # ノードを追加
            for term in term_list:
                # 所属する概念数をノードサイズに反映
                concept_count = len(term_to_concepts[term])
                G.add_node(term, size=10 + concept_count * 5, title=f"{term} (概念数: {concept_count})")

            # エッジを追加
            for i in range(len(term_list)):
                for j in range(i + 1, len(term_list)):
                    weight = cooccurrence_matrix[i, j]
                    if weight >= min_edge_weight:
                        G.add_edge(term_list[i], term_list[j], weight=float(weight), title=f"共起回数: {weight}")

            # PyVisを使用してインタラクティブなネットワーク図を作成
            net = Network(notebook=False, height="600px", width="100%", bgcolor="#ffffff", font_color="black")

            # NetworkXグラフをPyVisに変換
            net.from_nx(G)

            # ノードの色を設定
            for node in net.nodes:
                node["color"] = "#1f77b4"  # 青色

            # エッジの太さを重みに基づいて設定
            for edge in net.edges:
                # エッジの重みが存在する場合のみ設定
                if "weight" in edge:
                    edge["width"] = float(edge["weight"])
                else:
                    edge["width"] = 1.0

            # 物理シミュレーションの設定
            net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)

            # 一時ファイルにHTMLを保存
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                    net.save_graph(temp_file.name)

                    # HTMLファイルを読み込んでiframeで表示
                    with open(temp_file.name, 'r', encoding='utf-8') as f:
                        html_data = f.read()

                    st.components.v1.html(html_data, height=600)

                    # 一時ファイルを削除
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        # ファイル削除エラーを無視
                        pass
            except Exception as e:
                st.error(f"ネットワーク図の生成中にエラーが発生しました: {e}")
                st.info("別の可視化方法を試してみてください。")

            # 用語の所属概念情報を表示
            st.subheader("用語の所属概念")
            term_concept_data = []
            for term, concepts in term_to_concepts.items():
                term_concept_data.append({
                    "用語": term,
                    "所属概念": ", ".join(concepts),
                    "概念数": len(concepts)
                })

            term_concept_df = pd.DataFrame(term_concept_data)
            term_concept_df = term_concept_df.sort_values("概念数", ascending=False)
            st.dataframe(term_concept_df)

        # 概念-発言ネットワーク
        elif network_type == "概念-発言ネットワーク":
            st.subheader("概念と発言のネットワーク")
            st.write("概念と発言の関連性を可視化します。概念に該当する発言はエッジで結ばれます。")

            # 発言数の制限（パフォーマンス向上のため）
            max_utterances = st.slider("表示する最大発言数", 10, 100, 30,
                                       help="多すぎるとネットワークが複雑になります")

            # 概念に基づく発言の分析
            concept_matches = {}
            for concept_name, concept in st.session_state.concepts.items():
                matches = []
                for idx, row in st.session_state.data.iterrows():
                    text = str(row["発言内容"]).lower()

                    # 各ANDグループについて確認
                    all_and_conditions_met = True
                    for or_group in concept["and_groups"]:
                        any_or_condition_met = False
                        for term in or_group:
                            if term.lower() in text:
                                any_or_condition_met = True
                                break
                        if not any_or_condition_met:
                            all_and_conditions_met = False
                            break

                    if all_and_conditions_met:
                        matches.append(row["発言番号"])

                concept_matches[concept_name] = matches

            # NetworkXグラフを作成
            G = nx.Graph()

            # 概念ノードを追加
            for concept_name, matches in concept_matches.items():
                G.add_node(concept_name, size=20, group=1, title=concept_name, type="concept")

            # 発言ノードを追加（最大数を制限）
            utterance_count = min(max_utterances, len(st.session_state.data))
            for i in range(utterance_count):
                row = st.session_state.data.iloc[i]
                utterance_id = f"発言{row['発言番号']}"
                G.add_node(utterance_id, size=10, group=2, title=f"{utterance_id}: {row['発言内容'][:50]}...",
                           type="utterance")

            # エッジを追加
            for concept_name, matches in concept_matches.items():
                for utterance_num in matches:
                    # 表示する発言数の制限内かチェック
                    if any(st.session_state.data.iloc[i]["発言番号"] == utterance_num for i in range(utterance_count)):
                        utterance_id = f"発言{utterance_num}"
                        G.add_edge(concept_name, utterance_id, weight=1.0)

            # PyVisを使用してインタラクティブなネットワーク図を作成
            net = Network(notebook=False, height="600px", width="100%", bgcolor="#ffffff", font_color="black")

            # NetworkXグラフをPyVisに変換
            net.from_nx(G)

            # ノードの色とサイズを設定
            for node in net.nodes:
                if "type" in node and node["type"] == "concept":
                    node["color"] = "#ff7f0e"  # オレンジ色（概念）
                    node["size"] = 25
                else:
                    node["color"] = "#1f77b4"  # 青色（発言）
                    node["size"] = 15

            # 物理シミュレーションの設定
            net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)

            # 一時ファイルにHTMLを保存
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                    net.save_graph(temp_file.name)

                    # HTMLファイルを読み込んでiframeで表示
                    with open(temp_file.name, 'r', encoding='utf-8') as f:
                        html_data = f.read()

                    st.components.v1.html(html_data, height=600)

                    # 一時ファイルを削除
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        # ファイル削除エラーを無視
                        pass
            except Exception as e:
                st.error(f"ネットワーク図の生成中にエラーが発生しました: {e}")
                st.info("別の可視化方法を試してみてください。")

        # 発言者-概念ネットワーク
        elif network_type == "発言者-概念ネットワーク":
            st.subheader("発言者と概念のネットワーク")
            st.write(
                "発言者と概念の関連性を可視化します。発言者が特定の概念に関連する発言をした回数に基づいてエッジの太さが決まります。")

            # 最小エッジの重みを設定
            min_edge_weight = st.slider("最小発言回数", 1, 10, 1,
                                        help="発言者がこの回数以上言及した概念のみエッジを表示します")

            # 発言者ごとの概念出現回数を計算
            speaker_concept_counts = defaultdict(lambda: defaultdict(int))

            # 概念に基づく発言の分析
            for concept_name, concept in st.session_state.concepts.items():
                for idx, row in st.session_state.data.iterrows():
                    text = str(row["発言内容"]).lower()
                    speaker = row["発言者"]

                    # 各ANDグループについて確認
                    all_and_conditions_met = True
                    for or_group in concept["and_groups"]:
                        any_or_condition_met = False
                        for term in or_group:
                            if term.lower() in text:
                                any_or_condition_met = True
                                break
                        if not any_or_condition_met:
                            all_and_conditions_met = False
                            break

                    if all_and_conditions_met:
                        speaker_concept_counts[speaker][concept_name] += 1

            # NetworkXグラフを作成
            G = nx.Graph()

            # 発言者ノードを追加
            for speaker in speaker_concept_counts.keys():
                # 発言者の総発言数を計算
                total_utterances = sum(1 for _, row in st.session_state.data.iterrows() if row["発言者"] == speaker)
                G.add_node(speaker, size=15 + total_utterances / 2, group=1,
                           title=f"{speaker} (発言数: {total_utterances})", type="speaker")

            # 概念ノードを追加
            for concept_name in st.session_state.concepts.keys():
                # 概念に該当する発言数を計算
                concept_utterances = sum(counts[concept_name] for counts in speaker_concept_counts.values())
                G.add_node(concept_name, size=15 + concept_utterances, group=2,
                           title=f"{concept_name} (該当発言数: {concept_utterances})", type="concept")

            # エッジを追加
            for speaker, concepts in speaker_concept_counts.items():
                for concept_name, count in concepts.items():
                    if count >= min_edge_weight:
                        G.add_edge(speaker, concept_name, weight=float(count), title=f"発言回数: {count}")

            # PyVisを使用してインタラクティブなネットワーク図を作成
            net = Network(notebook=False, height="600px", width="100%", bgcolor="#ffffff", font_color="black")

            # NetworkXグラフをPyVisに変換
            net.from_nx(G)

            # ノードの色とサイズを設定
            for node in net.nodes:
                if "type" in node:
                    if node["type"] == "speaker":
                        node["color"] = "#2ca02c"  # 緑色（発言者）
                    elif node["type"] == "concept":
                        node["color"] = "#ff7f0e"  # オレンジ色（概念）

            # エッジの太さを重みに基づいて設定
            for edge in net.edges:
                # エッジの重みが存在する場合のみ設定
                if "weight" in edge:
                    edge["width"] = float(edge["weight"]) * 2
                else:
                    edge["width"] = 1.0

            # 物理シミュレーションの設定
            net.barnes_hut(gravity=-80000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)

            # 一時ファイルにHTMLを保存
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
                    net.save_graph(temp_file.name)

                    # HTMLファイルを読み込んでiframeで表示
                    with open(temp_file.name, 'r', encoding='utf-8') as f:
                        html_data = f.read()

                    st.components.v1.html(html_data, height=600)

                    # 一時ファイルを削除
                    try:
                        os.unlink(temp_file.name)
                    except Exception as e:
                        # ファイル削除エラーを無視
                        pass
            except Exception as e:
                st.error(f"ネットワーク図の生成中にエラーが発生しました: {e}")
                st.info("別の可視化方法を試してみてください。")

            # 発言者と概念の関連データを表示
            st.subheader("発言者と概念の関連データ")
            speaker_concept_data = []
            for speaker, concepts in speaker_concept_counts.items():
                for concept_name, count in concepts.items():
                    speaker_concept_data.append({
                        "発言者": speaker,
                        "概念": concept_name,
                        "発言回数": count
                    })

            if speaker_concept_data:
                speaker_concept_df = pd.DataFrame(speaker_concept_data)
                speaker_concept_df = speaker_concept_df.sort_values(["発言者", "発言回数"], ascending=[True, False])
                st.dataframe(speaker_concept_df)
            else:
                st.info("該当するデータがありません。")

        # 多次元尺度法(MDS)
        elif network_type == "多次元尺度法(MDS)":
            st.subheader("多次元尺度法(MDS)による概念・発言の可視化")
            st.write("概念と発言の関係性を2次元空間に配置します。近い位置にあるものは関連性が高いことを示します。")

            # 分析対象の選択
            mds_target = st.radio(
                "分析対象を選択してください",
                ["概念間の関係", "発言間の関係", "概念と発言の関係"]
            )

            if mds_target == "概念間の関係":
                # 概念間の類似度行列を作成
                concept_names = list(st.session_state.concepts.keys())

                if len(concept_names) < 2:
                    st.warning("MDSを実行するには少なくとも2つの概念が必要です。")
                else:
                    # 各概念に含まれる用語を収集
                    concept_terms = {}
                    for concept_name, concept in st.session_state.concepts.items():
                        terms = set()
                        for or_group in concept["and_groups"]:
                            terms.update(or_group)
                        concept_terms[concept_name] = terms

                    # Jaccard類似度行列を計算
                    n_concepts = len(concept_names)
                    similarity_matrix = np.zeros((n_concepts, n_concepts))

                    for i in range(n_concepts):
                        for j in range(n_concepts):
                            if i == j:
                                similarity_matrix[i, j] = 1.0
                            else:
                                terms_i = concept_terms[concept_names[i]]
                                terms_j = concept_terms[concept_names[j]]

                                # Jaccard類似度 = 共通要素数 / 和集合の要素数
                                intersection = len(terms_i.intersection(terms_j))
                                union = len(terms_i.union(terms_j))

                                if union > 0:
                                    similarity_matrix[i, j] = intersection / union

                    # 距離行列に変換 (1 - 類似度)
                    distance_matrix = 1 - similarity_matrix

                    # MDSを実行
                    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                    mds_result = mds.fit_transform(distance_matrix)

                    # Plotlyでインタラクティブな散布図を作成
                    fig = go.Figure()

                    # 概念をプロット
                    fig.add_trace(go.Scatter(
                        x=mds_result[:, 0],
                        y=mds_result[:, 1],
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color='#ff7f0e',
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        text=concept_names,
                        textposition="top center",
                        name='概念'
                    ))

                    # レイアウト設定
                    fig.update_layout(
                        title="概念間の関係性 (MDS)",
                        xaxis_title="次元1",
                        yaxis_title="次元2",
                        height=600,
                        width=800,
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # 類似度行列を表示
                    st.subheader("概念間の類似度行列")
                    similarity_df = pd.DataFrame(similarity_matrix, index=concept_names, columns=concept_names)
                    st.dataframe(similarity_df)

            elif mds_target == "発言間の関係":
                # 発言数の制限（パフォーマンス向上のため）
                max_utterances = st.slider("分析する最大発言数", 10, 100, 30,
                                           help="多すぎると計算に時間がかかります")

                # 発言を制限
                utterances = st.session_state.data.iloc[:max_utterances]

                # 発言と概念の関係を行列化
                utterance_concept_matrix = np.zeros((len(utterances), len(st.session_state.concepts)))

                for i, (_, row) in enumerate(utterances.iterrows()):
                    text = str(row["発言内容"]).lower()

                    for j, (concept_name, concept) in enumerate(st.session_state.concepts.items()):
                        # 各ANDグループについて確認
                        all_and_conditions_met = True
                        for or_group in concept["and_groups"]:
                            any_or_condition_met = False
                            for term in or_group:
                                if term.lower() in text:
                                    any_or_condition_met = True
                                    break
                            if not any_or_condition_met:
                                all_and_conditions_met = False
                                break

                        if all_and_conditions_met:
                            utterance_concept_matrix[i, j] = 1

                # 発言間のコサイン類似度を計算
                similarity_matrix = cosine_similarity(utterance_concept_matrix)

                # 距離行列に変換 (1 - 類似度)
                distance_matrix = 1 - similarity_matrix

                # MDSを実行
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                mds_result = mds.fit_transform(distance_matrix)

                # 発言番号のリスト
                utterance_ids = [f"発言{row['発言番号']}" for _, row in utterances.iterrows()]

                # 発言者情報
                speakers = [row['発言者'] for _, row in utterances.iterrows()]
                unique_speakers = list(set(speakers))

                # カラーマップの作成
                colors = px.colors.qualitative.Plotly[:len(unique_speakers)]
                speaker_color_map = {speaker: color for speaker, color in zip(unique_speakers, colors)}

                # Plotlyでインタラクティブな散布図を作成
                fig = go.Figure()

                # 発言者ごとに異なる色でプロット
                for speaker in unique_speakers:
                    indices = [i for i, s in enumerate(speakers) if s == speaker]

                    fig.add_trace(go.Scatter(
                        x=mds_result[indices, 0],
                        y=mds_result[indices, 1],
                        mode='markers+text',
                        marker=dict(
                            size=10,
                            color=speaker_color_map[speaker],
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        text=[utterance_ids[i] for i in indices],
                        textposition="top center",
                        name=speaker,
                        hovertext=[utterances.iloc[i]['発言内容'][:50] + "..." for i in indices]
                    ))

                # レイアウト設定
                fig.update_layout(
                    title="発言間の関係性 (MDS)",
                    xaxis_title="次元1",
                    yaxis_title="次元2",
                    height=600,
                    width=800,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

            else:  # 概念と発言の関係
                # 発言数の制限（パフォーマンス向上のため）
                max_utterances = st.slider("分析する最大発言数", 10, 50, 20,
                                           help="多すぎると計算に時間がかかります")

                # 発言を制限
                utterances = st.session_state.data.iloc[:max_utterances]

                # 概念名のリスト
                concept_names = list(st.session_state.concepts.keys())

                # 発言と概念の関係を行列化
                utterance_concept_matrix = np.zeros((len(utterances), len(concept_names)))

                for i, (_, row) in enumerate(utterances.iterrows()):
                    text = str(row["発言内容"]).lower()

                    for j, (concept_name, concept) in enumerate(st.session_state.concepts.items()):
                        # 各ANDグループについて確認
                        all_and_conditions_met = True
                        for or_group in concept["and_groups"]:
                            any_or_condition_met = False
                            for term in or_group:
                                if term.lower() in text:
                                    any_or_condition_met = True
                                    break
                            if not any_or_condition_met:
                                all_and_conditions_met = False
                                break

                        if all_and_conditions_met:
                            utterance_concept_matrix[i, j] = 1

                # 発言と概念を結合した行列を作成
                combined_matrix = np.zeros((len(utterances) + len(concept_names), len(concept_names)))
                combined_matrix[:len(utterances), :] = utterance_concept_matrix

                # 概念の部分は単位行列（自分自身との関連のみ1）
                for j in range(len(concept_names)):
                    combined_matrix[len(utterances) + j, j] = 1

                # コサイン類似度を計算
                similarity_matrix = cosine_similarity(combined_matrix)

                # 距離行列に変換 (1 - 類似度)
                distance_matrix = 1 - similarity_matrix

                # MDSを実行
                mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
                mds_result = mds.fit_transform(distance_matrix)

                # 発言と概念の座標を分離
                utterance_coords = mds_result[:len(utterances)]
                concept_coords = mds_result[len(utterances):]

                # 発言番号のリスト
                utterance_ids = [f"発言{row['発言番号']}" for _, row in utterances.iterrows()]

                # Plotlyでインタラクティブな散布図を作成
                fig = go.Figure()

                # 発言をプロット
                fig.add_trace(go.Scatter(
                    x=utterance_coords[:, 0],
                    y=utterance_coords[:, 1],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='#1f77b4',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=utterance_ids,
                    name='発言',
                    hovertext=[utterances.iloc[i]['発言内容'][:50] + "..." for i in range(len(utterances))]
                ))

                # 概念をプロット
                fig.add_trace(go.Scatter(
                    x=concept_coords[:, 0],
                    y=concept_coords[:, 1],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color='#ff7f0e',
                        symbol='diamond',
                        line=dict(width=1, color='DarkSlateGrey')
                    ),
                    text=concept_names,
                    textposition="top center",
                    name='概念'
                ))

                # レイアウト設定
                fig.update_layout(
                    title="概念と発言の関係性 (MDS)",
                    xaxis_title="次元1",
                    yaxis_title="次元2",
                    height=600,
                    width=800,
                    showlegend=True
                )

                st.plotly_chart(fig, use_container_width=True)

    else:
        if not st.session_state.data:
            st.info("CSVファイルをアップロードしてください。")
        elif not st.session_state.concepts:
            st.info("概念辞書を作成してください。")
