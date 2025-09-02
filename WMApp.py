import os
import streamlit as st
import json
from datetime import datetime
from uuid import uuid4

# Constants
DATA_FILE = "work_system.json"

# Initialize session state
def init_session_state():
    if 'projects' not in st.session_state:
        st.session_state.projects = []
    if 'selected_project_id' not in st.session_state:
        st.session_state.selected_project_id = None
    if 'show_completed' not in st.session_state:
        st.session_state.show_completed = True

class Task:
    def __init__(self, title, description="", due_date=None, priority="medium", status="todo"):
        self.id = str(uuid4())
        self.title = title
        self.description = description
        self.due_date = due_date
        self.priority = priority
        self.status = status
        self.created_at = datetime.now().isoformat()
        self.comments = []

    def add_comment(self, author, text):
        self.comments.append({
            "author": author,
            "text": text,
            "timestamp": datetime.now().isoformat()
        })

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "due_date": self.due_date,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "comments": self.comments
        }

class Project:
    def __init__(self, name, description=""):
        self.id = str(uuid4())
        self.name = name
        self.description = description
        self.created_at = datetime.now().isoformat()
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def get_task(self, task_id):
        return next((t for t in self.tasks if t.id == task_id), None)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
            "tasks": [task.to_dict() for task in self.tasks]
        }

def save_data():
    try:
        data = {
            "projects": [p.to_dict() for p in st.session_state.projects]
        }
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print("✅ Data saved")
    except Exception as e:
        st.error(f"Failed to save data: {str(e)}")

def load_data():
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    st.session_state.projects = []
                    return
                data = json.loads(content)

            st.session_state.projects = []
            for project_data in data.get("projects", []):
                project = Project(project_data["name"], project_data.get("description", ""))
                project.id = project_data["id"]
                project.created_at = project_data["created_at"]
                for task_data in project_data.get("tasks", []):
                    task = Task(
                        task_data["title"],
                        task_data.get("description", ""),
                        task_data.get("due_date", None),
                        task_data.get("priority", "medium"),
                        task_data.get("status", "todo")
                    )
                    task.id = task_data["id"]
                    task.created_at = task_data["created_at"]
                    task.comments = task_data.get("comments", [])
                    project.tasks.append(task)
                st.session_state.projects.append(project)
        else:
            st.session_state.projects = []
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        st.session_state.projects = []

def get_selected_project():
    if st.session_state.selected_project_id:
        for p in st.session_state.projects:
            if p.id == st.session_state.selected_project_id:
                return p
    return None

def create_project(name, description=""):
    project = Project(name, description)
    st.session_state.projects.append(project)
    save_data()
    st.session_state.selected_project_id = project.id
    return project

def create_task(project, title, description="", due_date=None, priority="medium"):
    task = Task(title, description, due_date, priority)
    project.add_task(task)
    save_data()
    return task

def delete_project(project_id):
    st.session_state.projects = [p for p in st.session_state.projects if p.id != project_id]
    if st.session_state.selected_project_id == project_id:
        st.session_state.selected_project_id = None
    save_data()

def delete_task(project, task_id):
    project.tasks = [t for t in project.tasks if t.id != task_id]
    save_data()

# Load session
init_session_state()
load_data()

# Main UI
st.title("🚀 Work Management System")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    menu = st.radio("Menu", ["Projects", "Create Project"])

    if st.session_state.selected_project_id:
        st.header("Filters")
        st.session_state.show_completed = st.checkbox("Show completed tasks", value=st.session_state.show_completed)

    if st.button("⚠️ Reset All Data"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
            st.success("All data reset.")
            st.rerun()

# Project List View
if menu == "Projects":
    st.header("Your Projects")

    if not st.session_state.projects:
        st.info("No projects found. Create one to get started!")

    for project in st.session_state.projects:
        with st.container():
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.subheader(project.name)
                st.caption(project.description)
                total = len(project.tasks)
                completed = sum(t.status == "done" for t in project.tasks)
                if total > 0:
                    st.progress(completed / total)
                    st.caption(f"{completed}/{total} tasks done")
                else:
                    st.caption("No tasks yet.")
            with col2:
                if st.button("Open", key=f"open_{project.id}"):
                    st.session_state.selected_project_id = project.id
                    st.rerun()
            with col3:
                if st.button("Delete", key=f"delete_{project.id}"):
                    delete_project(project.id)
                    st.rerun()
            st.divider()

# Create Project View
elif menu == "Create Project":
    st.header("Create New Project")
    with st.form("create_project_form"):
        name = st.text_input("Project Name", max_chars=50)
        description = st.text_area("Description", max_chars=200)

        if st.form_submit_button("Create Project"):
            if name.strip():
                create_project(name, description)
                st.success(f"Project '{name}' created!")
                st.balloons()
                st.rerun()
            else:
                st.error("Project name is required.")

# Project Detail View
selected_project = get_selected_project()
if selected_project:
    st.header(selected_project.name)
    st.caption(selected_project.description)

    if st.button("← Back to Projects"):
        st.session_state.selected_project_id = None
        st.rerun()

    tab1, tab2 = st.tabs(["📋 Tasks", "➕ Add New Task"])

    with tab1:
        visible_tasks = [t for t in selected_project.tasks if st.session_state.show_completed or t.status != "done"]
        if not visible_tasks:
            st.info("No tasks to display.")

        for task in visible_tasks:
            with st.expander(f"{task.title} - {task.status.upper()}"):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Description:** {task.description}")
                    st.markdown(f"**Priority:** {task.priority.capitalize()}")
                    if task.due_date:
                        st.markdown(f"**Due Date:** {task.due_date}")
                with col2:
                    if st.button("Delete", key=f"delete_{task.id}"):
                        delete_task(selected_project, task.id)
                        st.rerun()

                new_status = st.selectbox(
                    "Update Status",
                    ["todo", "in_progress", "done"],
                    index=["todo", "in_progress", "done"].index(task.status),
                    key=f"status_{task.id}"
                )
                if new_status != task.status:
                    task.status = new_status
                    save_data()
                    st.rerun()

                st.subheader("💬 Comments")
                if not task.comments:
                    st.info("No comments yet.")
                for comment in task.comments:
                    st.markdown(f"**{comment['author']}** ({comment['timestamp']}):")
                    st.markdown(f"> {comment['text']}")
                    st.divider()

                with st.form(key=f"comment_form_{task.id}"):
                    author = st.text_input("Your Name", key=f"author_{task.id}", max_chars=30)
                    text = st.text_area("Comment", key=f"text_{task.id}", max_chars=500)
                    if st.form_submit_button("Add Comment"):
                        if text:
                            task.add_comment(author or "Anonymous", text)
                            save_data()
                            st.rerun()

    with tab2:
        with st.form("add_task_form"):
            title = st.text_input("Task Title", max_chars=50)
            description = st.text_area("Description", max_chars=200)
            due_date = st.date_input("Due Date")
            priority = st.selectbox("Priority", ["low", "medium", "high"], index=1)

            if st.form_submit_button("Add Task"):
                if title.strip():
                    create_task(selected_project, title, description, str(due_date), priority)
                    st.success(f"Task '{title}' added!")
                    st.rerun()
                else:
                    st.error("Task title is required.")

# Custom styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
    }
    .st-expander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stProgress>div>div>div {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)
