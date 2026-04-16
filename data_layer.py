"""
SQLite-based data persistence layer for Chainlit.
Stores users (with bcrypt-hashed passwords), threads, steps, elements, and feedback.
"""

import json
import uuid
import aiosqlite
import bcrypt
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    Feedback,
    PageInfo,
    PaginatedResponse,
    Pagination,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User
from chainlit.step import StepDict
from chainlit.element import Element, ElementDict
from pathlib import Path

logger = logging.getLogger(__name__)

ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
DB_PATH = str(Path(__file__).parent / "storage" / "chainlit" / "chainlit_data.db")

class SQLiteDataLayer(BaseDataLayer):
    """Custom Chainlit data layer backed by SQLite with bcrypt user auth."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def _get_db(self) -> aiosqlite.Connection:
        if self._db is None:
            self._db = await aiosqlite.connect(self.db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._init_tables()
        return self._db

    async def _init_tables(self):
        db = self._db
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                identifier TEXT UNIQUE NOT NULL,
                display_name TEXT,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'USER',
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                user_identifier TEXT,
                name TEXT,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS steps (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                parent_id TEXT,
                type TEXT,
                name TEXT,
                command TEXT,
                input TEXT DEFAULT '',
                output TEXT DEFAULT '',
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                is_error INTEGER DEFAULT 0,
                created_at TEXT,
                start_time TEXT,
                end_time TEXT,
                generation TEXT,
                show_input TEXT,
                language TEXT,
                feedback_id TEXT
            );

            CREATE TABLE IF NOT EXISTS elements (
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                type TEXT,
                chainlit_key TEXT,
                url TEXT,
                object_key TEXT,
                name TEXT,
                display TEXT,
                size TEXT,
                language TEXT,
                page INTEGER,
                props TEXT,
                for_id TEXT,
                mime TEXT
            );

            CREATE TABLE IF NOT EXISTS feedbacks (
                id TEXT PRIMARY KEY,
                for_id TEXT NOT NULL,
                thread_id TEXT,
                value INTEGER NOT NULL,
                comment TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id);
            CREATE INDEX IF NOT EXISTS idx_threads_updated ON threads(updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps(thread_id);
            CREATE INDEX IF NOT EXISTS idx_elements_thread ON elements(thread_id);
            CREATE INDEX IF NOT EXISTS idx_feedbacks_for_id ON feedbacks(for_id);
        """)
        await db.commit()

    # ── User Management ──────────────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        db = await self._get_db()
        async with db.execute(
            "SELECT * FROM users WHERE identifier = ?", (identifier,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return PersistedUser(
                id=row["id"],
                identifier=row["identifier"],
                display_name=row["display_name"],
                metadata=json.loads(row["metadata"]),
                createdAt=row["created_at"],
            )

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        db = await self._get_db()
        # Check if user already exists
        existing = await self.get_user(user.identifier)
        if existing:
            return existing

        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).strftime(ISO_FORMAT)
        metadata = json.dumps(user.metadata)

        # Default password for users created via Chainlit (not via register_user)
        default_hash = bcrypt.hashpw(b"changeme", bcrypt.gensalt()).decode()

        await db.execute(
            """INSERT INTO users (id, identifier, display_name, password_hash, role, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, user.identifier, user.display_name, default_hash,
             user.metadata.get("role", "USER"), metadata, now),
        )
        await db.commit()

        return PersistedUser(
            id=user_id,
            identifier=user.identifier,
            display_name=user.display_name,
            metadata=user.metadata,
            createdAt=now,
        )

    async def register_user(self, identifier: str, password: str, role: str = "USER") -> PersistedUser:
        """Register a new user with a bcrypt-hashed password."""
        db = await self._get_db()

        # Check if user exists
        existing = await self.get_user(identifier)
        if existing:
            raise ValueError(f"User '{identifier}' already exists")

        user_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).strftime(ISO_FORMAT)
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        metadata = json.dumps({"role": role})

        await db.execute(
            """INSERT INTO users (id, identifier, display_name, password_hash, role, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, identifier, identifier, password_hash, role, metadata, now),
        )
        await db.commit()

        return PersistedUser(
            id=user_id,
            identifier=identifier,
            display_name=identifier,
            metadata={"role": role},
            createdAt=now,
        )

    async def verify_password(self, identifier: str, password: str) -> bool:
        """Verify a user's password against the stored bcrypt hash."""
        db = await self._get_db()
        async with db.execute(
            "SELECT password_hash FROM users WHERE identifier = ?", (identifier,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return False
            return bcrypt.checkpw(password.encode(), row["password_hash"].encode())

    async def get_user_role(self, identifier: str) -> Optional[str]:
        """Get the role of a user."""
        db = await self._get_db()
        async with db.execute(
            "SELECT role FROM users WHERE identifier = ?", (identifier,)
        ) as cursor:
            row = await cursor.fetchone()
            return row["role"] if row else None

    # ── Feedback ─────────────────────────────────────────────────────

    async def delete_feedback(self, feedback_id: str) -> bool:
        db = await self._get_db()
        await db.execute("DELETE FROM feedbacks WHERE id = ?", (feedback_id,))
        await db.commit()
        return True

    async def upsert_feedback(self, feedback: Feedback) -> str:
        db = await self._get_db()
        feedback_id = feedback.id or str(uuid.uuid4())
        await db.execute(
            """INSERT OR REPLACE INTO feedbacks (id, for_id, thread_id, value, comment)
               VALUES (?, ?, ?, ?, ?)""",
            (feedback_id, feedback.forId, feedback.threadId, feedback.value, feedback.comment),
        )
        await db.commit()
        return feedback_id

    # ── Feedback Reporting ────────────────────────────────────────────

    async def get_feedback_report(
        self,
        user_identifier: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Query feedback data with full context: timestamp, user question,
        assistant answer, feedback status, and comment.

        Joins: feedbacks → steps (the rated assistant message) → threads
        Then finds the preceding user_message step for the question.

        Returns a list of dicts:
            {
                "timestamp": str,
                "user": str,
                "question": str,
                "answer": str,
                "status": "liked" | "not liked",
                "value": int,
                "comment": str | None,
                "thread_id": str,
            }
        """
        db = await self._get_db()

        # Get all feedback records with their associated steps and threads
        query = """
            SELECT
                f.id AS feedback_id,
                f.for_id,
                f.thread_id,
                f.value,
                f.comment,
                s.output AS answer,
                s.created_at AS answer_timestamp,
                s.type AS step_type,
                t.user_identifier
            FROM feedbacks f
            LEFT JOIN steps s ON s.id = f.for_id
            LEFT JOIN threads t ON t.id = f.thread_id
        """
        params = []

        if user_identifier:
            query += " WHERE t.user_identifier = ?"
            params.append(user_identifier)

        query += " ORDER BY s.created_at DESC LIMIT ?"
        params.append(limit)

        results = []
        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        for row in rows:
            thread_id = row["thread_id"]
            for_id = row["for_id"]
            answer_timestamp = row["answer_timestamp"] or ""

            # Find the preceding user message in the same thread
            question = ""
            if thread_id and for_id:
                q = """
                    SELECT output FROM steps
                    WHERE thread_id = ?
                      AND type = 'user_message'
                      AND created_at <= ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """
                async with db.execute(q, (thread_id, answer_timestamp)) as q_cursor:
                    q_row = await q_cursor.fetchone()
                    if q_row:
                        question = q_row["output"] or ""

            # Map feedback value to human-readable status
            value = row["value"]
            if value == 1:
                status = "liked"
            elif value == 0:
                status = "not liked"
            else:
                status = f"value={value}"

            results.append({
                "timestamp": answer_timestamp,
                "user": row["user_identifier"] or "unknown",
                "question": question,
                "answer": (row["answer"] or "")[:500],  # Truncate long answers
                "status": status,
                "value": value,
                "comment": row["comment"],
                "thread_id": thread_id or "",
                "feedback_id": row["feedback_id"],
            })

        return results

    # ── Elements ─────────────────────────────────────────────────────

    async def create_element(self, element: Element):
        db = await self._get_db()
        element_dict = element.to_dict()
        await db.execute(
            """INSERT OR REPLACE INTO elements
               (id, thread_id, type, chainlit_key, url, object_key, name, display, size, language, page, props, for_id, mime)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                element_dict.get("id"),
                element_dict.get("threadId"),
                element_dict.get("type"),
                element_dict.get("chainlitKey"),
                element_dict.get("url"),
                element_dict.get("objectKey"),
                element_dict.get("name"),
                element_dict.get("display"),
                element_dict.get("size"),
                element_dict.get("language"),
                element_dict.get("page"),
                json.dumps(element_dict.get("props")) if element_dict.get("props") else None,
                element_dict.get("forId"),
                element_dict.get("mime"),
            ),
        )
        await db.commit()

    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        db = await self._get_db()
        async with db.execute(
            "SELECT * FROM elements WHERE id = ? AND thread_id = ?",
            (element_id, thread_id),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return ElementDict(
                id=row["id"],
                threadId=row["thread_id"],
                type=row["type"],
                chainlitKey=row["chainlit_key"],
                url=row["url"],
                objectKey=row["object_key"],
                name=row["name"],
                display=row["display"],
                size=row["size"],
                language=row["language"],
                page=row["page"],
                props=json.loads(row["props"]) if row["props"] else None,
                forId=row["for_id"],
                mime=row["mime"],
            )

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        db = await self._get_db()
        if thread_id:
            await db.execute(
                "DELETE FROM elements WHERE id = ? AND thread_id = ?",
                (element_id, thread_id),
            )
        else:
            await db.execute("DELETE FROM elements WHERE id = ?", (element_id,))
        await db.commit()

    # ── Steps ────────────────────────────────────────────────────────

    async def create_step(self, step_dict: StepDict):
        db = await self._get_db()
        thread_id = step_dict.get("threadId")
        
        # Auto-create thread if it doesn't exist yet
        if thread_id:
            async with db.execute(
                "SELECT id FROM threads WHERE id = ?", (thread_id,)
            ) as cursor:
                if not await cursor.fetchone():
                    now = datetime.now(timezone.utc).strftime(ISO_FORMAT)
                    await db.execute(
                        """INSERT OR IGNORE INTO threads (id, name, created_at, updated_at)
                           VALUES (?, ?, ?, ?)""",
                        (thread_id, "New Chat", now, now),
                    )
        
        await db.execute(
            """INSERT OR REPLACE INTO steps
               (id, thread_id, parent_id, type, name, command, input, output, metadata, tags,
                is_error, created_at, start_time, end_time, generation, show_input, language)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                step_dict.get("id"),
                step_dict.get("threadId"),
                step_dict.get("parentId"),
                step_dict.get("type"),
                step_dict.get("name"),
                step_dict.get("command"),
                step_dict.get("input", ""),
                step_dict.get("output", ""),
                json.dumps(step_dict.get("metadata", {})),
                json.dumps(step_dict.get("tags", [])),
                1 if step_dict.get("isError") else 0,
                step_dict.get("createdAt"),
                step_dict.get("start"),
                step_dict.get("end"),
                json.dumps(step_dict.get("generation")) if step_dict.get("generation") else None,
                str(step_dict.get("showInput")) if step_dict.get("showInput") is not None else None,
                step_dict.get("language"),
            ),
        )
        await db.commit()

    async def update_step(self, step_dict: StepDict):
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        db = await self._get_db()
        await db.execute("DELETE FROM steps WHERE id = ?", (step_id,))
        await db.commit()

    # ── Threads ──────────────────────────────────────────────────────

    async def get_thread_author(self, thread_id: str) -> str:
        db = await self._get_db()
        async with db.execute(
            "SELECT user_identifier FROM threads WHERE id = ?", (thread_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return row["user_identifier"] if row and row["user_identifier"] else ""

    async def delete_thread(self, thread_id: str):
        db = await self._get_db()
        await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        await db.commit()

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        db = await self._get_db()

        query = "SELECT * FROM threads WHERE 1=1"
        params = []

        if filters.userId:
            query += " AND user_id = ?"
            params.append(filters.userId)

        if filters.search:
            query += " AND name LIKE ?"
            params.append(f"%{filters.search}%")

        if pagination.cursor:
            query += " AND updated_at < ?"
            params.append(pagination.cursor)

        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(pagination.first + 1)  # +1 to check hasNextPage

        async with db.execute(query, params) as cursor:
            rows = await cursor.fetchall()

        has_next = len(rows) > pagination.first
        threads_rows = rows[: pagination.first]

        threads: List[ThreadDict] = []
        for row in threads_rows:
            thread_id = row["id"]

            # Fetch steps for this thread
            async with db.execute(
                "SELECT * FROM steps WHERE thread_id = ? ORDER BY created_at ASC",
                (thread_id,),
            ) as step_cursor:
                step_rows = await step_cursor.fetchall()

            steps = []
            for s in step_rows:
                feedback_dict = None
                if s["feedback_id"]:
                    async with db.execute(
                        "SELECT * FROM feedbacks WHERE id = ?", (s["feedback_id"],)
                    ) as fb_cursor:
                        fb_row = await fb_cursor.fetchone()
                        if fb_row:
                            feedback_dict = {
                                "id": fb_row["id"],
                                "forId": fb_row["for_id"],
                                "value": fb_row["value"],
                                "comment": fb_row["comment"],
                            }

                step: StepDict = {
                    "id": s["id"],
                    "threadId": s["thread_id"],
                    "parentId": s["parent_id"],
                    "type": s["type"],
                    "name": s["name"],
                    "command": s["command"],
                    "input": s["input"] or "",
                    "output": s["output"] or "",
                    "metadata": json.loads(s["metadata"]) if s["metadata"] else {},
                    "tags": json.loads(s["tags"]) if s["tags"] else [],
                    "isError": bool(s["is_error"]),
                    "createdAt": s["created_at"],
                    "start": s["start_time"],
                    "end": s["end_time"],
                    "generation": json.loads(s["generation"]) if s["generation"] else None,
                    "showInput": s["show_input"],
                    "language": s["language"],
                    "feedback": feedback_dict,
                }
                steps.append(step)

            # Fetch elements
            async with db.execute(
                "SELECT * FROM elements WHERE thread_id = ?", (thread_id,)
            ) as el_cursor:
                el_rows = await el_cursor.fetchall()

            elements = []
            for e in el_rows:
                el: ElementDict = {
                    "id": e["id"],
                    "threadId": e["thread_id"],
                    "type": e["type"],
                    "chainlitKey": e["chainlit_key"],
                    "url": e["url"],
                    "objectKey": e["object_key"],
                    "name": e["name"],
                    "display": e["display"],
                    "size": e["size"],
                    "language": e["language"],
                    "page": e["page"],
                    "props": json.loads(e["props"]) if e["props"] else None,
                    "forId": e["for_id"],
                    "mime": e["mime"],
                }
                elements.append(el)

            thread: ThreadDict = {
                "id": thread_id,
                "createdAt": row["created_at"],
                "name": row["name"],
                "userId": row["user_id"],
                "userIdentifier": row["user_identifier"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
                "steps": steps,
                "elements": elements,
            }
            threads.append(thread)

        page_info = PageInfo(
            hasNextPage=has_next,
            startCursor=threads_rows[0]["updated_at"] if threads_rows else None,
            endCursor=threads_rows[-1]["updated_at"] if threads_rows else None,
        )

        return PaginatedResponse(pageInfo=page_info, data=threads)

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        db = await self._get_db()
        async with db.execute(
            "SELECT * FROM threads WHERE id = ?", (thread_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None

        # Fetch steps
        async with db.execute(
            "SELECT * FROM steps WHERE thread_id = ? ORDER BY created_at ASC",
            (thread_id,),
        ) as step_cursor:
            step_rows = await step_cursor.fetchall()

        steps = []
        for s in step_rows:
            step: StepDict = {
                "id": s["id"],
                "threadId": s["thread_id"],
                "parentId": s["parent_id"],
                "type": s["type"],
                "name": s["name"],
                "input": s["input"] or "",
                "output": s["output"] or "",
                "metadata": json.loads(s["metadata"]) if s["metadata"] else {},
                "createdAt": s["created_at"],
                "start": s["start_time"],
                "end": s["end_time"],
            }
            steps.append(step)

        # Fetch elements
        async with db.execute(
            "SELECT * FROM elements WHERE thread_id = ?", (thread_id,)
        ) as el_cursor:
            el_rows = await el_cursor.fetchall()

        elements = []
        for e in el_rows:
            el: ElementDict = {
                "id": e["id"],
                "threadId": e["thread_id"],
                "type": e["type"],
                "url": e["url"],
                "name": e["name"],
                "display": e["display"],
                "mime": e["mime"],
            }
            elements.append(el)

        return ThreadDict(
            id=thread_id,
            createdAt=row["created_at"],
            name=row["name"],
            userId=row["user_id"],
            userIdentifier=row["user_identifier"],
            tags=json.loads(row["tags"]) if row["tags"] else [],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            steps=steps,
            elements=elements,
        )

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        db = await self._get_db()
        now = datetime.now(timezone.utc).strftime(ISO_FORMAT)

        # Check if thread exists
        async with db.execute(
            "SELECT id FROM threads WHERE id = ?", (thread_id,)
        ) as cursor:
            exists = await cursor.fetchone()

        if not exists:
            # Get user_identifier from user_id
            user_identifier = None
            if user_id:
                async with db.execute(
                    "SELECT identifier FROM users WHERE id = ?", (user_id,)
                ) as cursor:
                    user_row = await cursor.fetchone()
                    user_identifier = user_row["identifier"] if user_row else None

            await db.execute(
                """INSERT INTO threads (id, user_id, user_identifier, name, metadata, tags, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    user_id,
                    user_identifier,
                    name,
                    json.dumps(metadata or {}),
                    json.dumps(tags or []),
                    now,
                    now,
                ),
            )
        else:
            updates = ["updated_at = ?"]
            params = [now]

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if user_id is not None:
                updates.append("user_id = ?")
                params.append(user_id)
                # Also update user_identifier
                async with db.execute(
                    "SELECT identifier FROM users WHERE id = ?", (user_id,)
                ) as cursor:
                    user_row = await cursor.fetchone()
                    if user_row:
                        updates.append("user_identifier = ?")
                        params.append(user_row["identifier"])
            if metadata is not None:
                updates.append("metadata = ?")
                params.append(json.dumps(metadata))
            if tags is not None:
                updates.append("tags = ?")
                params.append(json.dumps(tags))

            params.append(thread_id)
            await db.execute(
                f"UPDATE threads SET {', '.join(updates)} WHERE id = ?", params
            )

        await db.commit()

    # ── Debug ────────────────────────────────────────────────────────

    async def build_debug_url(self) -> str:
        return ""

    # ── Cleanup ──────────────────────────────────────────────────────

    async def close(self):
        if self._db:
            await self._db.close()
            self._db = None


from config import INITIAL_USERS

async def seed_default_users(data_layer: SQLiteDataLayer):
    """Seed default users if they don't exist, or update their passwords if they do."""
    db = await data_layer._get_db()
    for identifier, password, role in INITIAL_USERS:
        try:
            # We use a manual insert with ON CONFLICT to ensure even existing users 
            # have their passwords updated to the latest 'seed' password.
            password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            user_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).strftime(ISO_FORMAT)
            metadata = json.dumps({"role": role})

            await db.execute(
                """INSERT INTO users (id, identifier, display_name, password_hash, role, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(identifier) DO UPDATE SET password_hash=excluded.password_hash""",
                (user_id, identifier, identifier, password_hash, role, metadata, now),
            )
            await db.commit()
            logger.info(f"Seeded/Updated user: {identifier}")
        except Exception as e:
            logger.error(f"Failed to seed user {identifier}: {e}")
