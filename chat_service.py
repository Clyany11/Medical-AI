from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from datetime import datetime
import requests
import json
import os
import uuid
from openai import OpenAI
import time
from typing import Optional
from dotenv import load_dotenv
# 本模块在 app.py 的 load_dotenv() 之前被导入, 必须自行加载 .env, 否则读不到 DB_* 配置
load_dotenv()
# 路由实例
router = APIRouter(
    tags=["chat_history"]
)
# =======================================
# 数据库配置 (从 .env 读取)
# =======================================
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "aidoctor")
DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)
SessionLocal = sessionmaker(bind=engine)
# =======================================
# DeepSeek 模型配置
# =======================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_URL = f"{os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')}/v1/chat/completions"
# =======================================
# 请求模型
# =======================================
class SessionListRequest(BaseModel):
    username: str
class SessionMessageRequest(BaseModel):
    username: str
    session_id: str
class ChatRequest(BaseModel):
    username: Optional[str] = "anonymous"
    session_id: Optional[str] = None
    question: str
# =======================================
# 获取数据库连接
# =======================================
def get_db():
    db = SessionLocal()
    return db
# =======================================
# 获取用户所有会话
# =======================================
@router.post("/get_session_list")
def get_session_list(req: SessionListRequest):
    db = get_db()
    try:
        sql = text(
            """
            SELECT
                id,
                topic,
                started_at,
                update_time
            FROM user_chat_session
            WHERE username=:username
            AND is_deleted=0
            ORDER BY update_time DESC
            """
        )
        result = db.execute(sql, {
            "username": req.username
        })
        rows = result.fetchall()
        data = []
        for row in rows:
            data.append({
                "session_id": row[0],
                "topic": row[1],
                "started_at": str(row[2]) if row[2] else None,
                "update_time": str(row[3]) if row[3] else None
            })
        return {
            "code": 0,
            "data": data
        }
    finally:
        db.close()
# =======================================
# 获取某个会话全部消息
# =======================================
@router.post("/get_session_messages")
def get_session_messages(req: SessionMessageRequest):
    db = get_db()
    try:
        check_sql = text(
            """
            SELECT id
            FROM user_chat_session
            WHERE id=:session_id
            AND username=:username
            AND is_deleted=0
            """
        )
        check = db.execute(check_sql, {
            "session_id": req.session_id,
            "username": req.username
        }).fetchone()
        if not check:
            return {
                "code": 1,
                "msg": "session not exists"
            }
        sql = text(
            """
            SELECT
                sender,
                message_text,
                message_timestamp
            FROM user_chat_message
            WHERE session_id=:session_id
            AND is_deleted=0
            ORDER BY message_timestamp ASC
            """
        )
        result = db.execute(sql, {
            "session_id": req.session_id
        })
        rows = result.fetchall()
        messages = []
        for row in rows:
            messages.append({
                "sender": row[0],
                "message": row[1],
                "time": str(row[2]) if row[2] else None
            })
        return {
            "code": 0,
            "data": messages
        }
    finally:
        db.close()
# =======================================
# 创建session
# =======================================
def create_session(db, username, first_message, session_id):
    topic = first_message[:30]
    # 前端没传 session_id 时生成唯一 id
    # 注意: 这里绝不能用固定值 (如 "abc"), 否则所有"新会话"会共用同一个主键 id,
    #       第二个新会话 INSERT 时主键冲突 → 建会话失败 → 该会话不出现在左侧历史列表,
    #       且不同用户的首轮消息会全部串进同一条会话. 必须用全局唯一的 uuid.
    if not session_id:
        session_id = str(uuid.uuid4())
    sql = text(
        """
        INSERT INTO user_chat_session
        (
            id,
            username,
            started_at,
            topic,
            create_time,
            update_time,
            create_by,
            update_by,
            is_deleted
        )
        VALUES
        (
            :id,
            :username,
            :started_at,
            :topic,
            :create_time,
            :update_time,
            :create_by,
            :update_by,
            0
        )
        """
    )
    now = datetime.now()
    db.execute(sql, {
        "id": session_id,
        "username": username,
        "started_at": now,
        "topic": topic,
        "create_time": now,
        "update_time": now,
        "create_by": username,
        "update_by": username
    })
    db.commit()
    return session_id
# =======================================
# 保存消息
# =======================================
def save_message(db, session_id, sender, message_text, username):
    sql = text(
        """
        INSERT INTO user_chat_message
        (
            session_id,
            sender,
            message_text,
            message_timestamp,
            create_time,
            update_time,
            create_by,
            update_by,
            is_deleted
        )
        VALUES
        (
            :session_id,
            :sender,
            :message_text,
            :message_timestamp,
            :create_time,
            :update_time,
            :create_by,
            :update_by,
            0
        )
        """
    )
    now = datetime.now()
    db.execute(sql, {
        "session_id": session_id,
        "sender": sender,
        "message_text": message_text,
        "message_timestamp": now,
        "create_time": now,
        "update_time": now,
        "create_by": username,
        "update_by": username
    })
    update_sql = text(
        """
        UPDATE user_chat_session
        SET update_time=:update_time
        WHERE id=:session_id
        """
    )
    db.execute(update_sql, {
        "update_time": now,
        "session_id": session_id
    })
    db.commit()
# =======================================
# 查询session是否存在
# =======================================
def check_session_exists(db, username, session_id):
    sql = text(
        """
        SELECT id
        FROM user_chat_session
        WHERE id=:session_id
        AND username=:username
        AND is_deleted=0
        """
    )
    result = db.execute(sql, {
        "session_id": session_id,
        "username": username
    }).fetchone()
    return result
# =======================================
# 获取历史上下文
# =======================================
def get_history_messages(db, session_id):
    sql = text(
        """
        SELECT sender, message_text
        FROM user_chat_message
        WHERE session_id=:session_id
        AND is_deleted=0
        ORDER BY message_timestamp ASC
        LIMIT 20
        """
    )
    rows = db.execute(sql, {
        "session_id": session_id
    }).fetchall()
    history = []
    for row in rows:
        role = "assistant"
        if row[0] == "user":
            role = "user"
        history.append({
            "role": role,
            "content": row[1]
        })
    return history