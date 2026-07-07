from flask_login import UserMixin

from extensions import db


class AdminUser(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)


class AgentSettings(db.Model):
    """Singleton (id=1): admin-only LLM selection for the support agent."""

    __tablename__ = "agent_settings"

    id = db.Column(db.Integer, primary_key=True)
    llm_model_key = db.Column(db.String(128), nullable=False, default="auto")
