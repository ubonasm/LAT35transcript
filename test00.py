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

# 言語モデルの辞書
language_models = {
    "ja": "ja_core_news_sm",
    "en": "en_core_web_sm",
    "zh": "zh_core_web_sm",
    "ko": "ko_core_news_sm",
    "fr": "fr_core_news_sm"
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


# 言語モデルのロード
@st.cache_resource
def load_nlp_model(lang):
    try:
        return spacy.load(language_models[lang])
    except OSError:
        st.warning(f"{language_options[lang]}モデルをダウンロードしています...")
        spacy.cli.download(language_models[lang])
        return spacy.load(language_models[lang])


if selected_language != st.session_state.language:
    st.session_state.language = selected_language
    st.session_state.nlp = load_nlp_model(selected_language)
elif st.session_state.nlp is None:
    st.session_state.nlp = load_nlp_model(selected_language)

# ファイルアップロード
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        # 必要なカラムがあるか確認
        required_columns = ["発言番号", "発言者", "発言内容"]
        if not all(col in data.columns for col in required_columns):
            st.error("CSVファイルには「発言番号」「発言者」「発言内容」の列が必要です。")
        else:
            st.session_state.data = data
            st.sidebar.success("ファイルが正常に読み込まれました。")
    except Exception as e:
        st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# タブの作成
tab1, tab2, tab3 = st.tabs(["KWIC分析", "概念辞書作成", "概念分析・可視化"])

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
uploaded_concepts = st.sidebar.file_uploader("概念辞書をインポート", type=["json"])
if uploaded_concepts is not None:
    try:
        imported_concepts = json.load(uploaded_concepts)
        st.session_state.concepts.update(imported_concepts)
        st.sidebar.success("概念辞書がインポートされました。")
    except Exception as e:
        st.sidebar.error(f"インポート中にエラーが発生しました: {e}")