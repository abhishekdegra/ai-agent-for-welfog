import click
from flask import abort, flash, redirect, render_template, request, url_for
from flask.cli import with_appcontext
from flask_login import login_required, login_user, logout_user
from werkzeug.security import check_password_hash, generate_password_hash

from admin_models import AdminUser
from services.agent_llm_settings import (
    describe_auto_chain,
    get_admin_llm_model_key,
    get_selectable_llm_options,
    option_for_key,
    set_admin_llm_model_key,
)
from extensions import db
from services.kb_service import get_allowed_knowledge_filenames, refresh_knowledge_cache
from services.knowledge_reindex_service import (
    sync_admin_txt_create,
    sync_admin_txt_delete,
    sync_admin_txt_update,
)
from services.mysql_service import get_knowledge_document_by_title
from utils.validators import _safe_knowledge_filename


def register_admin_routes(app):
    @app.route("/welfog-admin-login", methods=["GET", "POST"])
    def admin_login():
        if request.method == "POST":
            u = (request.form.get("username") or "").strip()
            p = request.form.get("password") or ""
            user = AdminUser.query.filter_by(username=u).first()

            if user:
                valid = check_password_hash(user.password, p) if user.password.startswith("pbkdf2:") else (user.password == p)
                if valid:
                    if not user.password.startswith("pbkdf2:"):
                        user.password = generate_password_hash(p, method="pbkdf2:sha256")
                        db.session.commit()
                    login_user(user)
                    return redirect(url_for("admin_dashboard"))

            flash("Invalid username or password.", "error")
        return render_template("admin_login.html")

    @app.route("/welfog-admin")
    @login_required
    def admin_dashboard():
        return redirect(url_for("admin_knowledge"))

    @app.route("/welfog-admin/knowledge")
    @login_required
    def admin_knowledge():
        files = get_allowed_knowledge_filenames()
        return render_template("admin_knowledge.html", files=files, nav_active="knowledge")

    @app.route("/welfog-admin/settings", methods=["GET", "POST"])
    @login_required
    def admin_settings():
        if request.method == "POST":
            model_key = (request.form.get("llm_model_key") or "").strip()
            ok, msg = set_admin_llm_model_key(model_key)
            flash(msg, "success" if ok else "error")
            return redirect(url_for("admin_settings"))

        llm_key = get_admin_llm_model_key()
        llm_current = option_for_key(llm_key) or option_for_key("auto")
        return render_template(
            "admin_settings.html",
            nav_active="settings",
            knowledge_storage="MySQL knowledge_documents + Qdrant",
            llm_model_key=llm_key,
            llm_current=llm_current,
            llm_options=get_selectable_llm_options(),
            auto_chain_hint=describe_auto_chain(),
        )

    @app.route("/welfog-admin/new-file", methods=["GET", "POST"])
    @login_required
    def admin_add_file():
        if request.method == "POST":
            raw_name = (request.form.get("name") or "").strip()
            initial_content = request.form.get("content") or ""
            safe_name = _safe_knowledge_filename(raw_name)
            if not safe_name:
                flash("Invalid filename. Use letters, numbers, underscore or hyphen.", "error")
                return render_template("admin_new_file.html", nav_active="knowledge")

            filename = f"{safe_name}.txt"
            if get_knowledge_document_by_title(safe_name):
                flash("Document already exists. Choose a different name.", "error")
                return render_template("admin_new_file.html", nav_active="knowledge")

            sync_result = sync_admin_txt_create(safe_name, initial_content.strip())
            refresh_knowledge_cache()
            if not sync_result.get("ok"):
                flash(
                    f"Created {filename} in MySQL but knowledge re-index failed: "
                    f"{sync_result.get('reindex', {}).get('error') or sync_result.get('error')}",
                    "error",
                )
            else:
                flash(f"Created {filename} and indexed to knowledge pipeline.", "success")
            return redirect(url_for("admin_knowledge"))
        return render_template("admin_new_file.html", nav_active="knowledge")

    @app.route("/welfog-admin/edit/<filename>", methods=["GET", "POST"])
    @login_required
    def edit_kb_file(filename):
        allowed_files = set(get_allowed_knowledge_filenames())
        if filename not in allowed_files:
            abort(404)
        title = filename[:-4] if filename.lower().endswith(".txt") else filename
        doc = get_knowledge_document_by_title(title)
        if not doc:
            abort(404)
        if request.method == "POST":
            new_content = request.form["content"]
            sync_result = sync_admin_txt_update(title, new_content)
            refresh_knowledge_cache()
            if not sync_result.get("ok"):
                flash(
                    f"Saved {filename} in MySQL but knowledge re-index failed: "
                    f"{sync_result.get('reindex', {}).get('error') or sync_result.get('error')}",
                    "error",
                )
            else:
                flash(f"Saved {filename} and re-indexed knowledge pipeline.", "success")
            return redirect(url_for("admin_knowledge"))

        content = doc.get("content") or ""
        return render_template("admin_edit.html", filename=filename, content=content, nav_active="knowledge")

    @app.route("/welfog-admin/delete/<filename>", methods=["POST"])
    @login_required
    def delete_kb_file(filename):
        allowed_files = set(get_allowed_knowledge_filenames())
        if filename not in allowed_files:
            abort(404)
        title = filename[:-4] if filename.lower().endswith(".txt") else filename
        sync_admin_txt_delete(title)
        refresh_knowledge_cache()
        flash(f"Deleted {filename} and removed from knowledge pipeline.", "success")
        return redirect(url_for("admin_knowledge"))

    @app.route("/admin-logout")
    @login_required
    def admin_logout():
        logout_user()
        return redirect(url_for("admin_login"))

    @app.cli.command("create-admin")
    @click.argument("username")
    @click.argument("password")
    @with_appcontext
    def create_admin(username, password):
        db.create_all()
        username = (username or "").strip()
        if len(username) < 3:
            print("Username must be at least 3 chars.")
            return
        if len(password or "") < 8:
            print("Password must be at least 8 chars.")
            return
        existing = AdminUser.query.filter_by(username=username).first()
        if existing:
            print(f"Admin '{username}' already exists.")
            return
        new_admin = AdminUser(username=username, password=generate_password_hash(password, method="pbkdf2:sha256"))
        db.session.add(new_admin)
        db.session.commit()
        print(f"Admin '{username}' created successfully.")
