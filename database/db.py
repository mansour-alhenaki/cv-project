"""
database/db.py
--------------
SQLite database للسجلات والتنبيهات
"""

from sqlalchemy import (
    create_engine, Column, String,
    Boolean, DateTime, Float, Text
)
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime
import uuid
import hashlib

Base = declarative_base()
engine = create_engine(
    "sqlite:///safety.db",
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ─────────────────────────────────────────
# جدول 1: سجل السلامة
# ─────────────────────────────────────────
class SafetyLog(Base):
    __tablename__ = "safety_logs"

    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id  = Column(String, nullable=False)
    timestamp    = Column(DateTime, default=datetime.datetime.now)

    helmet       = Column(Boolean, default=False)
    vest         = Column(Boolean, default=False)
    gloves       = Column(Boolean, default=False)
    goggles      = Column(Boolean, default=False)
    boots        = Column(Boolean, default=False)
    ppe_compliant= Column(Boolean, default=False)

    near_ladder    = Column(Boolean, default=False)
    ladder_angle   = Column(Float,   nullable=True)
    ladder_zone    = Column(String,  nullable=True)
    three_point_ok = Column(Boolean, nullable=True)
    contact_count  = Column(String,  nullable=True)

    alert_sent   = Column(Boolean, default=False)
    alert_msg    = Column(Text,    nullable=True)


# ─────────────────────────────────────────
# جدول 2: بيانات الموظفين
# ─────────────────────────────────────────
class Employee(Base):
    __tablename__ = "employees"

    employee_id   = Column(String, primary_key=True)
    name          = Column(String, nullable=True)
    department    = Column(String, nullable=True)
    active        = Column(Boolean, default=True)
    password_hash = Column(String, nullable=True)


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    Base.metadata.create_all(engine)

    # أضف عمود password_hash إذا لم يكن موجوداً (migration)
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns("employees")]
    if "password_hash" not in cols:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE employees ADD COLUMN password_hash VARCHAR"))
            conn.commit()

    db = SessionLocal()
    samples = [
        Employee(employee_id="EMP-001", name="أحمد محمد",   department="الإنتاج", password_hash=_hash("1234")),
        Employee(employee_id="EMP-002", name="خالد علي",    department="الصيانة", password_hash=_hash("1234")),
        Employee(employee_id="EMP-003", name="سعد عبدالله", department="الأمن",   password_hash=_hash("1234")),
    ]
    for emp in samples:
        existing = db.query(Employee).filter_by(employee_id=emp.employee_id).first()
        if not existing:
            db.add(emp)
        elif not existing.password_hash:
            existing.password_hash = _hash("1234")
    db.commit()
    db.close()
    print("✅ Database initialized")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def register_employee(employee_id: str, name: str, department: str, password: str) -> dict:
    db = SessionLocal()
    try:
        if db.query(Employee).filter_by(employee_id=employee_id.upper()).first():
            return {"success": False, "error": "id_exists"}
        emp = Employee(
            employee_id=employee_id.upper(),
            name=name,
            department=department,
            active=True,
            password_hash=_hash(password),
        )
        db.add(emp)
        db.commit()
        return {"success": True, "employee_id": emp.employee_id, "name": emp.name, "department": emp.department}
    finally:
        db.close()


def verify_login(employee_id: str, password: str) -> dict:
    db = SessionLocal()
    try:
        emp = db.query(Employee).filter_by(employee_id=employee_id.upper(), active=True).first()
        if not emp or emp.password_hash != _hash(password):
            return {"success": False, "error": "invalid_credentials"}
        return {"success": True, "employee_id": emp.employee_id, "name": emp.name, "department": emp.department}
    finally:
        db.close()


def save_log(log_data: dict):
    db = SessionLocal()
    try:
        log = SafetyLog(**log_data)
        db.add(log)
        db.commit()
        db.refresh(log)
        return log.id
    finally:
        db.close()


def get_recent_logs(limit: int = 50):
    db = SessionLocal()
    try:
        logs = (
            db.query(SafetyLog)
            .order_by(SafetyLog.timestamp.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id":            l.id,
                "employee_id":   l.employee_id,
                "timestamp":     str(l.timestamp),
                "ppe_compliant": l.ppe_compliant,
                "missing_ppe":   _get_missing(l),
                "near_ladder":   l.near_ladder,
                "ladder_zone":   l.ladder_zone,
                "three_point":   l.three_point_ok,
                "alert_msg":     l.alert_msg,
            }
            for l in logs
        ]
    finally:
        db.close()


def get_employee_info(employee_id: str):
    db = SessionLocal()
    try:
        emp = db.query(Employee).filter_by(employee_id=employee_id).first()
        if emp:
            return {"employee_id": emp.employee_id, "name": emp.name, "department": emp.department}
        return {"employee_id": employee_id, "name": "غير مسجل", "department": "-"}
    finally:
        db.close()


def _get_missing(log: SafetyLog):
    missing = []
    if not log.helmet:  missing.append("helmet")
    if not log.vest:    missing.append("vest")
    if not log.gloves:  missing.append("gloves")
    if not log.goggles: missing.append("goggles")
    if not log.boots:   missing.append("boots")
    return missing
