import streamlit as st
from drive_manager import DriveManager
from ai_agent import GeminiDriveAgent

# Настройка мобильного интерфейса
st.set_page_config(
    page_title="Drive AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Функция инициализации менеджера диска
@st.cache_resource(show_spinner=False)
def get_drive_manager():
    if "gcp_service_account" not in st.secrets:
        st.error("В st.secrets не найден ключ [gcp_service_account]. Добавьте его в настройках.")
        st.stop()
    return DriveManager(dict(st.secrets["gcp_service_account"]))

# Кэширование сканирования метаданных (TTL 2 часа)
@st.cache_data(ttl=7200, show_spinner="Синхронизация индекса с Google Диска...")
def load_photos_index(_drive_mgr):
    return _drive_mgr.get_all_photos_metadata()

# Проверка API-ключа Gemini
if "GEMINI_API_KEY" not in st.secrets:
    st.error("В st.secrets не найден GEMINI_API_KEY. Добавьте его в настройках.")
    st.stop()

# Инициализация сервисов
drive_mgr = get_drive_manager()

# Боковая панель (настройки и статус)
with st.sidebar:
    st.header("⚙️ Статус и настройки")
    
    model_choice = st.selectbox(
        "Модель Gemini",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Flash работает быстрее, Pro — глубже анализирует сложные детали и схемы."
    )
    
    photos_index = load_photos_index(drive_mgr)
    st.success(f"Проиндексировано фото: **{len(photos_index)}**")
    
    if st.button("🔄 Обновить индекс файлов", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Инициализация ИИ-агента
@st.cache_resource(show_spinner=False)
def get_ai_agent(api_key: str, model_name: str):
    return GeminiDriveAgent(api_key=api_key, drive_manager=drive_mgr, model_name=model_name)

agent = get_ai_agent(st.secrets["GEMINI_API_KEY"], model_choice)
agent.set_photos_index(photos_index)

# Инициализация истории чата в сессии
if "messages" not in st.session_state:
    st.session_state.messages = []

# Заголовок страницы
st.title("🧠 Персональный Drive-ассистент")
st.caption("Поиск по вашим фото, счетам, схемам и метаданным на Google Диске")

# Отображение истории сообщений
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "images" in msg and msg["images"]:
            cols = st.columns(len(msg["images"]))
            for idx, img_bytes in enumerate(msg["images"]):
                cols[idx].image(img_bytes, caption="Просмотрено моделью", use_container_width=True)

# Поле ввода запроса
if prompt := st.chat_input("Спросите о документах, покупках, ремонте или фото..."):
    # Добавляем сообщение пользователя в интерфейс и историю
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Генерация ответа ассистента
    with st.chat_message("assistant"):
        with st.spinner("Анализирую метаданные и фотографии..."):
            try:
                # Передаем историю диалога в агент
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                
                response_text, _, loaded_images = agent.generate_response(history, prompt)
                
                # Выводим текстовый ответ
                st.markdown(response_text)
                
                # Если модель открывала оригиналы фото — показываем их пользователю
                if loaded_images:
                    cols = st.columns(len(loaded_images))
                    for idx, img_bytes in enumerate(loaded_images):
                        cols[idx].image(img_bytes, caption="Изученный оригинал", use_container_width=True)

                # Сохраняем в историю сессии
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "images": loaded_images
                })

            except Exception as e:
                st.error(f"Произошла ошибка при обработке запроса: {e}")