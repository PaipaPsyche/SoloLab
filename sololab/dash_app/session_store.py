"""Server-side storage for per-session data that can't live in a dcc.Store
(numpy arrays, pandas DataFrames, raw file bytes) because it isn't JSON
serializable. Backed by diskcache so it works across multiple gunicorn
workers, unlike a plain in-process dict.

Each browser tab gets a session_id (uuid4) kept in a dcc.Store with
storage_type="session" (see app.py). Values are addressed as
(session_id, name) pairs; see constants.py for the canonical `name`
strings used across the app (e.g. "stix_counts_final").
"""
import os

import diskcache

DEFAULT_CACHE_DIR = os.environ.get(
    "SOLOLAB_SESSION_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), ".session_cache"),
)
DEFAULT_TTL_SECONDS = int(os.environ.get("SOLOLAB_SESSION_TTL", 6 * 3600))


class SessionStore:
    def __init__(self, cache_dir=DEFAULT_CACHE_DIR, ttl=DEFAULT_TTL_SECONDS):
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = diskcache.Cache(cache_dir)
        self.ttl = ttl

    @staticmethod
    def _key(session_id, name):
        return f"{session_id}:{name}"

    def set(self, session_id, name, value):
        self._cache.set(self._key(session_id, name), value, expire=self.ttl)

    def get(self, session_id, name, default=None):
        return self._cache.get(self._key(session_id, name), default=default)

    def has(self, session_id, name):
        return self._key(session_id, name) in self._cache

    def touch(self, session_id, name):
        self._cache.touch(self._key(session_id, name), expire=self.ttl)

    def delete(self, session_id, name):
        self._cache.delete(self._key(session_id, name))

    def clear_session(self, session_id, names):
        for name in names:
            self.delete(session_id, name)

    def evict_stale(self):
        """Force a full sweep of expired entries. diskcache normally expires
        entries lazily on access, so call this periodically (e.g. on worker
        startup) on a long-running server to avoid accumulating stale files."""
        return self._cache.expire()


session_store = SessionStore()
