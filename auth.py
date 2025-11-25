from __future__ import annotations
import os
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any
import streamlit as st
import toml

APP_DIR = Path(__file__).resolve().parent

DEFAULT_CRED_PATH = APP_DIR / "credentials.toml"
CREDENTIALS_PATH = Path(
    os.environ.get("ANAHEALTH_CREDENTIALS_PATH", str(DEFAULT_CRED_PATH))
)


@st.cache_resource(show_spinner=False)
def load_users() -> Dict[str, Dict[str, Any]]:
    """
    Lê o arquivo credentials.toml e devolve um dict:
    { email: {email, password_hash, role, id} }
    """
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de credenciais não encontrado em: {CREDENTIALS_PATH}"
        )

    data = toml.load(CREDENTIALS_PATH)
    users_section = data.get("users", {})

    users_by_email: Dict[str, Dict[str, Any]] = {}

    for user_id, info in users_section.items():
        email = info.get("email")
        password = info.get("password")
        role = info.get("role", "user")

        if not email or not password:
            continue

        pwd_hash = hashlib.sha256(password.encode()).hexdigest()

        users_by_email[email] = {
            "id": user_id,
            "email": email,
            "password_hash": pwd_hash,
            "role": role,
        }

    return users_by_email


def authenticate(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Confere se email/senha batem com algum usuário do TOML.
    Retorna o dict do usuário se der certo; caso contrário, None.
    """
    users = load_users()
    user = users.get(email)
    if not user:
        return None

    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    if pwd_hash != user["password_hash"]:
        return None

    return user


def _login_form():
    """Renderiza o formulário de login."""
    st.title("🔐 Login - Ana Health")
    st.write("Acesso restrito. Informe seu e-mail e senha para entrar.")

    email = st.text_input("E-mail", key="login_email")
    password = st.text_input("Senha", type="password", key="login_password")

    if st.button("Entrar", type="primary"):
        user = authenticate(email, password)
        if user:
            st.session_state["authenticated"] = True
            st.session_state["user"] = user
            st.success("Login realizado com sucesso!")
            st.rerun()
        else:
            st.error("E-mail ou senha inválidos. Verifique e tente novamente.")


def ensure_authenticated(min_role: str | None = None) -> Dict[str, Any]:
    """
    Garante que o usuário está autenticado.
    - Se não estiver: mostra o formulário de login e dá st.stop()
    - Se estiver: retorna o dict do usuário atual

    min_role (opcional): se você quiser exigir perfil mínimo:
      "viewer" < "manager" < "admin"
    """
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["user"] = None

    if not st.session_state["authenticated"]:
        _login_form()
        st.stop()

    user = st.session_state.get("user") or {}
    user_role = user.get("role", "user")

    # Se não for exigida role mínima, só devolve
    if not min_role:
        return user

    role_order = {"viewer": 1, "user": 1, "manager": 2, "admin": 3}
    if role_order.get(user_role, 0) < role_order.get(min_role, 0):
        st.error("Você não tem permissão para acessar esta área.")
        st.stop()

    return user


def get_current_user() -> Optional[Dict[str, Any]]:
    """Atalho para pegar o usuário atual da sessão."""
    return st.session_state.get("user")


def logout_button(sidebar: bool = True, label: str = "Sair"):
    """
    Cria um botão de logout (na sidebar por padrão).
    """
    container = st.sidebar if sidebar else st
    if container.button(label):
        st.session_state["authenticated"] = False
        st.session_state["user"] = None
        st.rerun()