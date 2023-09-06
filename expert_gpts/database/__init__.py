import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, scoped_session, sessionmaker

engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///var/db.sqlite"))


def get_session():
    session_factory = sessionmaker(bind=engine)
    return scoped_session(session_factory)


@contextmanager
def get_db_session():
    session = get_session()
    try:
        yield session
        session.close()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.remove()


def with_db_session(func):
    def wrapper(*args, **kwargs):
        with get_db_session() as session:
            return func(session, *args, **kwargs)

    return wrapper


class Base(DeclarativeBase):
    pass
