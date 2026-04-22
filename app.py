import streamlit as st
import requests
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="МФЦ RAG Система",
    page_icon="🏢",
    layout="wide"
)

st.title("🏢 Система вопрос-ответ МФЦ")
st.markdown("""
Загрузите документы (txt, pdf, md) в базу знаний и задавайте вопросы по регламентам МФЦ.
""")

# Sidebar for navigation
menu = st.sidebar.selectbox("Меню", ["Задать вопрос", "Загрузить документы", "Статус системы"])

if menu == "Задать вопрос":
    st.header("❓ Задать вопрос")
    
    question = st.text_area(
        "Введите ваш вопрос:",
        placeholder="Например: Какие документы нужны для получения паспорта?",
        height=100
    )
    
    if st.button("Получить ответ", type="primary"):
        if not question.strip():
            st.warning("Пожалуйста, введите вопрос.")
        else:
            with st.spinner("Генерирую ответ..."):
                try:
                    response = requests.post(
                        f"{BACKEND_URL}/ask",
                        json={"question": question},
                        timeout=120
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        st.success("✅ Ответ получен!")
                        st.markdown("### Ответ:")
                        st.write(data["answer"])
                        
                        if data.get("sources"):
                            st.markdown("### Источники:")
                            for source in data["sources"]:
                                st.info(f"📄 {source}")
                        
                        if not data.get("context_used"):
                            st.warning("⚠️ Ответ сгенерирован без использования базы знаний.")
                    
                    else:
                        st.error(f"Ошибка: {response.status_code} - {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("Не удалось подключиться к серверу. Проверьте, запущен ли backend.")
                except requests.exceptions.Timeout:
                    st.error("Превышено время ожидания ответа. Попробуйте еще раз.")
                except Exception as e:
                    st.error(f"Произошла ошибка: {str(e)}")

elif menu == "Загрузить документы":
    st.header("📁 Загрузка документов")
    
    uploaded_files = st.file_uploader(
        "Выберите файлы для загрузки",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True
    )
    
    if st.button("Загрузить в базу знаний", type="primary"):
        if not uploaded_files:
            st.warning("Пожалуйста, выберите хотя бы один файл.")
        else:
            with st.spinner("Загружаю документы..."):
                try:
                    files = []
                    for uploaded_file in uploaded_files:
                        files.append(("files", (uploaded_file.name, uploaded_file.getvalue())))
                    
                    response = requests.post(
                        f"{BACKEND_URL}/ingest",
                        files=files,
                        timeout=300
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"✅ Загрузка завершена! Добавлено чанков: {data['chunks_added']}")
                        
                        if data.get("errors"):
                            st.warning("Ошибки при загрузке:")
                            for error in data["errors"]:
                                st.error(error)
                    else:
                        st.error(f"Ошибка: {response.status_code} - {response.text}")
                
                except requests.exceptions.ConnectionError:
                    st.error("Не удалось подключиться к серверу. Проверьте, запущен ли backend.")
                except requests.exceptions.Timeout:
                    st.error("Превышено время ожидания. Большие файлы могут загружаться долго.")
                except Exception as e:
                    st.error(f"Произошла ошибка: {str(e)}")

elif menu == "Статус системы":
    st.header("📊 Статус системы")
    
    if st.button("Обновить статус"):
        try:
            # Health check
            health_response = requests.get(f"{BACKEND_URL}/health", timeout=10)
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success("✅ Backend доступен")
                st.json(health_data)
            
            # Stats
            stats_response = requests.get(f"{BACKEND_URL}/stats", timeout=10)
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                st.info(f"📈 Векторов в базе: {stats_data.get('vectors_count', 0)}")
                st.json(stats_data)
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Не удалось подключиться к серверу.")
        except Exception as e:
            st.error(f"Ошибка: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
МФЦ RAG Система | Powered by Ollama + Qdrant + LangChain
</div>
""", unsafe_allow_html=True)
