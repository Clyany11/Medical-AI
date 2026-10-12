"""用户认证模块: 注册 / 登录 / JWT 校验.

- 用户数据存 MySQL `aidoctor.users`
- 密码用 bcrypt 哈希
- 登录成功签发 JWT, 后续请求通过 Authorization: Bearer <token> 携带
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import text

# 本模块可能先于 app.py 的 load_dotenv() 被导入, 自行加载 .env
load_dotenv()

# 复用 chat_service 的连接池, 避免重复创建 engine
from chat_service import SessionLocal

# ===================== 配置 =====================
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-aidoctor-dev-secret")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


# ===================== 请求模型 =====================
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=6, max_length=64)
    nickname: Optional[str] = Field(None, max_length=20)


class LoginRequest(BaseModel):
    username: str
    password: str


# ===================== 密码工具 =====================
def _hash_password(password: str) -> str:
    # bcrypt 只处理前 72 字节, 超出部分截断避免报错
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        pw_bytes = password.encode("utf-8")[:72]
        return bcrypt.checkpw(pw_bytes, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# ===================== JWT 工具 =====================
def _create_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": now,
        "exp": now + timedelta(minutes=JWT_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="登录已过期, 请重新登录")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="无效的登录凭证")


# ===================== 依赖: 取当前用户 =====================
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """从 Authorization: Bearer <token> 解析当前用户, 失败抛 401."""
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="未提供登录凭证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = _decode_token(credentials.credentials)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="无效的登录凭证")
    return {"user_id": user_id, "username": payload.get("username")}


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """可选鉴权: 无 token 或 token 无效时返回 None, 不抛异常."""
    if credentials is None or not credentials.credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None


def resolve_user_from_authorization(authorization: Optional[str]) -> Optional[dict]:
    """从原始 Authorization 请求头解析用户, 供非 Depends 场景使用 (如 /chat_stream).

    :param authorization: 形如 "Bearer <token>" 的请求头, 可为 None
    :return: {"user_id": str, "username": str} 或 None (无 token / 无效)
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization[7:].strip()
    try:
        payload = _decode_token(token)
    except HTTPException:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return {"user_id": user_id, "username": payload.get("username")}


# ===================== 接口 =====================
@router.post("/register")
def register(req: RegisterRequest):
    db = SessionLocal()
    try:
        exists = db.execute(
            text("SELECT id FROM users WHERE username = :u"),
            {"u": req.username},
        ).first()
        if exists:
            raise HTTPException(status.HTTP_409_CONFLICT, detail="用户名已存在")

        nickname = req.nickname or req.username
        nickname_taken = db.execute(
            text("SELECT id FROM users WHERE nickname = :n"),
            {"n": nickname},
        ).first()
        if nickname_taken:
            raise HTTPException(status.HTTP_409_CONFLICT, detail="昵称已被占用")

        db.execute(
            text(
                "INSERT INTO users (username, password_hash, nickname) "
                "VALUES (:u, :p, :n)"
            ),
            {"u": req.username, "p": _hash_password(req.password), "n": nickname},
        )
        db.commit()

        user_id = db.execute(
            text("SELECT id FROM users WHERE username = :u"),
            {"u": req.username},
        ).scalar()
        return {
            "code": 0,
            "message": "注册成功",
            "access_token": _create_token(user_id, req.username),
            "token_type": "bearer",
            "user": {"id": user_id, "username": req.username, "nickname": nickname},
        }
    finally:
        db.close()


@router.post("/login")
def login(req: LoginRequest):
    db = SessionLocal()
    try:
        row = db.execute(
            text(
                "SELECT id, username, password_hash, nickname "
                "FROM users WHERE username = :u"
            ),
            {"u": req.username},
        ).first()
    finally:
        db.close()

    if row is None or not _verify_password(req.password, row.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    return {
        "code": 0,
        "message": "登录成功",
        "access_token": _create_token(row.id, row.username),
        "token_type": "bearer",
        "user": {"id": row.id, "username": row.username, "nickname": row.nickname},
    }


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)):
    db = SessionLocal()
    try:
        row = db.execute(
            text(
                "SELECT id, username, nickname, created_at "
                "FROM users WHERE id = :i"
            ),
            {"i": current_user["user_id"]},
        ).first()
    finally:
        db.close()

    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return {
        "code": 0,
        "user": {
            "id": row.id,
            "username": row.username,
            "nickname": row.nickname,
            "created_at": str(row.created_at),
        },
    }
