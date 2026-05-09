from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from customer_service_agent.models import Complaint, Customer, CustomerMemory, Order


def _order_to_dict(order: Order) -> dict[str, Any]:
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "product_name": order.product_name,
        "status": order.status,
        "order_date": order.order_date.isoformat() if order.order_date else None,
        "delivery_date": order.delivery_date.isoformat() if order.delivery_date else None,
    }


def _customer_to_dict(customer: Customer) -> dict[str, Any]:
    return {
        "customer_id": customer.customer_id,
        "name": customer.name,
        "email": customer.email,
        "created_at": customer.created_at.isoformat() if customer.created_at else None,
    }


class CustomerServiceRepository:
    def __init__(self, session_factory: sessionmaker[Session]):
        self._session_factory = session_factory

    def get_order(self, order_id: int) -> dict[str, Any] | None:
        with self._session_factory() as session:
            order = session.get(Order, order_id)
            if not order:
                return None
            return _order_to_dict(order)

    def get_customer(self, customer_id: int) -> dict[str, Any] | None:
        with self._session_factory() as session:
            customer = session.get(Customer, customer_id)
            if not customer:
                return None
            return _customer_to_dict(customer)

    def request_refund(self, order_id: int) -> dict[str, Any] | None:
        with self._session_factory() as session:
            order = session.get(Order, order_id)
            if not order:
                return None
            order.status = "refund_requested"
            session.commit()
            session.refresh(order)
            return _order_to_dict(order)

    def cancel_order(self, order_id: int) -> dict[str, Any] | None:
        with self._session_factory() as session:
            order = session.get(Order, order_id)
            if not order:
                return None
            order.status = "cancel_requested"
            session.commit()
            session.refresh(order)
            return _order_to_dict(order)

    def log_complaint(
        self,
        customer_id: int,
        issue: str,
        order_id: int | None = None,
        status: str = "open",
    ) -> dict[str, Any]:
        with self._session_factory() as session:
            complaint = Complaint(
                customer_id=customer_id,
                order_id=order_id,
                issue=issue,
                status=status,
            )
            session.add(complaint)
            session.commit()
            session.refresh(complaint)
            return {
                "complaint_id": complaint.complaint_id,
                "customer_id": complaint.customer_id,
                "order_id": complaint.order_id,
                "issue": complaint.issue,
                "status": complaint.status,
                "created_at": complaint.created_at.isoformat(),
            }

    def write_memory(self, customer_id: int, key: str, value: str) -> dict[str, Any]:
        with self._session_factory() as session:
            memory = CustomerMemory(customer_id=customer_id, key=key, value=value)
            session.add(memory)
            session.commit()
            session.refresh(memory)
            return {
                "id": memory.id,
                "customer_id": memory.customer_id,
                "key": memory.key,
                "value": memory.value,
                "created_at": memory.created_at.isoformat(),
            }

    def read_memories(self, customer_id: int, key: str | None = None) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = select(CustomerMemory).where(CustomerMemory.customer_id == customer_id)
            if key:
                stmt = stmt.where(CustomerMemory.key == key)
            stmt = stmt.order_by(CustomerMemory.created_at.desc())
            memories = session.scalars(stmt).all()
            return [
                {
                    "id": memory.id,
                    "customer_id": memory.customer_id,
                    "key": memory.key,
                    "value": memory.value,
                    "created_at": memory.created_at.isoformat(),
                }
                for memory in memories
            ]

    def list_complaints(self, customer_id: int) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = (
                select(Complaint)
                .where(Complaint.customer_id == customer_id)
                .order_by(Complaint.created_at.desc())
            )
            complaints = session.scalars(stmt).all()
            return [
                {
                    "complaint_id": complaint.complaint_id,
                    "customer_id": complaint.customer_id,
                    "order_id": complaint.order_id,
                    "issue": complaint.issue,
                    "status": complaint.status,
                    "created_at": complaint.created_at.isoformat(),
                }
                for complaint in complaints
            ]

    def summarize_issue_patterns(self, customer_id: int) -> dict[str, Any]:
        complaints = self.list_complaints(customer_id)
        normalized = []
        for complaint in complaints:
            issue = complaint["issue"].lower()
            if "late" in issue or "delay" in issue:
                normalized.append("late_delivery")
            elif "refund" in issue:
                normalized.append("refund")
            else:
                normalized.append("other")
        counts = Counter(normalized)
        return {
            "total_complaints": len(complaints),
            "issue_counts": dict(counts),
            "repeated_late_delivery": counts["late_delivery"] >= 2,
        }

