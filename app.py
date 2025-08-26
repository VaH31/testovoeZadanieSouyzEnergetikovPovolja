import os
import requests
import streamlit as st
from transformers import pipeline
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
load_dotenv()

DetectorFactory.seed = 0

@st.cache_resource
def load_en_pipeline():
    model_id = "cardiffnlp/twitter-roberta-base-sentiment"
    return pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)


@st.cache_resource
def load_ru_pipeline():
    model_id = "blanchefort/rubert-base-cased-sentiment"
    return pipeline("sentiment-analysis", model=model_id, tokenizer=model_id)


def detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return 'ru' if lang.startswith('ru') else 'en'
    except Exception:
        return 'en'


def map_label(label: str, lang: str) -> str:
    l = label.upper()
    if lang == 'en':
        mapping = {
            'LABEL_0': 'NEGATIVE',
            'LABEL_1': 'NEUTRAL',
            'LABEL_2': 'POSITIVE',
            'NEGATIVE': 'NEGATIVE',
            'NEUTRAL': 'NEUTRAL',
            'POSITIVE': 'POSITIVE',
        }
        return mapping.get(l, l)
    else:
        return l if l in ['POSITIVE', 'NEUTRAL', 'NEGATIVE'] else l


def analyze_sentiment(text: str):
    lang = detect_lang(text)
    clf = load_ru_pipeline() if lang == 'ru' else load_en_pipeline()
    chunk = text.strip()[:2000]  # ограничим для устойчивости
    res = clf(chunk)[0]
    label = map_label(res['label'], lang)
    score = float(res.get('score', 0.0))
    return {'tone': label, 'score': score, 'lang': lang}


# --- Mistral API  ---
def mistral_generate_recs(transcript: str, tone_ui: str):
    api_key = "os.getenv(checkKey)"
    if not api_key:
        return None, "MISTRAL_API_KEY not set"

    endpoint = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "mistral-tiny",  # или "mistral-small"/"mistral-medium"
        "messages": [
            {"role": "system", "content": "Ты — помощник по качеству звонков."},
            {"role": "user", "content": f"Тон разговора: {tone_ui}.\nТекст: {transcript}\n\nДай 1–2 краткие рекомендации по улучшению."}
        ],
        "max_tokens": 120
    }
    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip(), None
    except Exception as e:
        return None, str(e)


def rule_based_recs(tone: str, lang: str) -> str:
    if lang == 'ru':
        if tone == 'NEGATIVE':
            return "- Спокойно признать проблему и извиниться.\n- Задавать уточняющие вопросы и предложить конкретный следующий шаг с дедлайном."
        if tone == 'NEUTRAL':
            return "- Добавить эмпатию (подтверждать, что услышали).\n- Чётко структурировать шаги: что мы сделаем и когда."
        return "- Подтвердить договорённости письмом.\n- Спросить, что можно улучшить в следующем контакте."
    else:
        if tone == 'NEGATIVE':
            return "- Acknowledge the issue and apologize.\n- Ask clarifying questions and offer a concrete next step with a timeline."
        if tone == 'NEUTRAL':
            return "- Add empathy cues (reflect and confirm).\n- Structure the next steps clearly: what, who, when."
        return "- Confirm agreements in writing.\n- Ask for feedback on how to improve the next interaction."


st.set_page_config(page_title="📞 Агент оценки звонков", page_icon="📞", layout="centered")
st.title("📞 Агент оценки звонков")
st.caption("MVP: определение тона + рекомендации. Локальная sentiment-модель + Mistral API для генерации рекомендаций.")


with st.form("input_form"):
    text = st.text_area("Вставьте расшифровку разговора", height=200, placeholder="Клиент: ... Оператор: ...")
    submitted = st.form_submit_button("Оценить")


if submitted:
    if not text or not text.strip():
        st.warning("Введите текст разговора.")
    else:
        with st.spinner("Анализирую..."):
            result = analyze_sentiment(text)
        tone_map = {'NEGATIVE': 'негативный', 'NEUTRAL': 'нейтральный', 'POSITIVE': 'положительный'}
        ui_tone = tone_map.get(result['tone'], result['tone'])

        st.subheader("Результат")
        st.write(f"**Тон общения:** {ui_tone} \n**Уверенность:** {result['score']:.2f} \n**Определённый язык:** {result['lang']}")

        # Попытка LLM через Mistral; если нет — фоллбек
        gen, err = mistral_generate_recs(text, ui_tone)
        if gen:
            st.write("**Рекомендации:**")
            st.write(gen)
        else:
            st.write("**Рекомендации (правило-базовый фоллбек):**")
            st.write(rule_based_recs(result['tone'], result['lang']))
        if err:
            with st.expander("Техническая справка"):
                st.code(str(err))

        st.divider()
        with st.expander("Отладочная информация"):
            st.json(result)
