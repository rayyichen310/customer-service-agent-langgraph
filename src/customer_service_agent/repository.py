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


def _complaint_to_dict(complaint: Complaint) -> dict[str, Any]:
    return {
        "complaint_id": complaint.complaint_id,
        "customer_id": complaint.customer_id,
        "order_id": complaint.order_id,
        "issue": complaint.issue,
        "status": complaint.status,
        "created_at": complaint.created_at.isoformat(),
    }


def _memory_to_dict(memory: CustomerMemory) -> dict[str, Any]:
    return {
        "id": memory.id,
        "customer_id": memory.customer_id,
        "key": memory.key,
        "value": memory.value,
        "created_at": memory.created_at.isoformat(),
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

    def get_order_for_customer(self, order_id: int, customer_id: int) -> dict[str, Any] | None:
        with self._session_factory() as session:
            stmt = select(Order).where(
                Order.order_id == order_id,
                Order.customer_id == customer_id,
            )
            order = session.scalars(stmt).first()
            if not order:
                return None
            return _order_to_dict(order)

    def list_orders_for_customer(self, customer_id: int) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = (
                select(Order)
                .where(Order.customer_id == customer_id)
                .order_by(Order.order_date.desc(), Order.order_id.desc())
            )
            orders = session.scalars(stmt).all()
            return [_order_to_dict(order) for order in orders]

    def get_customer(self, customer_id: int) -> dict[str, Any] | None:
        with self._session_factory() as session:
            customer = session.get(Customer, customer_id)
            if not customer:
                return None
            return _customer_to_dict(customer)

    def request_refund(self, order_id: int, customer_id: int | None = None) -> dict[str, Any] | None:
        return self._update_order_status(order_id, "refund_requested", customer_id)

    def cancel_order(self, order_id: int, customer_id: int | None = None) -> dict[str, Any] | None:
        return self._update_order_status(order_id, "cancel_requested", customer_id)

    def _update_order_status(
        self,
        order_id: int,
        status: str,
        customer_id: int | None = None,
    ) -> dict[str, Any] | None:
        with self._session_factory() as session:
            order = session.get(Order, order_id)
            if not order or (customer_id is not None and order.customer_id != customer_id):
                return None
            order.status = status
            session.commit()
            session.refresh(order)
            return _order_to_dict(order)

    def log_complaint(
        self,
        customer_id: int,
        issue: str,
        order_id: int | None = None,
        status: str = "open",
    ) -> dict[str, Any] | None:
        with self._session_factory() as session:
            if order_id is not None:
                order = session.get(Order, order_id)
                if not order or order.customer_id != customer_id:
                    return None
            complaint = Complaint(
                customer_id=customer_id,
                order_id=order_id,
                issue=issue,
                status=status,
            )
            session.add(complaint)
            session.commit()
            session.refresh(complaint)
            return _complaint_to_dict(complaint)

    def write_memory(self, customer_id: int, key: str, value: str) -> dict[str, Any]:
        with self._session_factory() as session:
            stmt = select(CustomerMemory).where(
                CustomerMemory.customer_id == customer_id,
                CustomerMemory.key == key,
            )
            memory = session.scalars(stmt).first()
            if memory:
                memory.value = value
            else:
                memory = CustomerMemory(customer_id=customer_id, key=key, value=value)
                session.add(memory)
            session.commit()
            session.refresh(memory)
            return _memory_to_dict(memory)

    def read_memories(self, customer_id: int, key: str | None = None) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = select(CustomerMemory).where(CustomerMemory.customer_id == customer_id)
            if key:
                stmt = stmt.where(CustomerMemory.key == key)
            stmt = stmt.order_by(CustomerMemory.created_at.desc())
            memories = session.scalars(stmt).all()
            return [_memory_to_dict(memory) for memory in memories]

    def list_complaints(self, customer_id: int) -> list[dict[str, Any]]:
        with self._session_factory() as session:
            stmt = (
                select(Complaint)
                .where(Complaint.customer_id == customer_id)
                .order_by(Complaint.created_at.desc())
            )
            complaints = session.scalars(stmt).all()
            return [_complaint_to_dict(complaint) for complaint in complaints]

    def summarize_issue_patterns(self, customer_id: int) -> dict[str, Any]:
        complaints = self.list_complaints(customer_id)
        counts = Counter(complaint["issue"].strip().lower() for complaint in complaints)
        return {
            "total_complaints": len(complaints),
            "issue_counts": dict(counts),
            "repeated_issues": {
                issue: count for issue, count in counts.items() if count > 1
            },
        }

    def customer_snapshot(self, customer_id: int) -> dict[str, Any]:
        return {
            "customer": self.get_customer(customer_id),
            "orders": self.list_orders_for_customer(customer_id),
            "complaints": self.list_complaints(customer_id),
            "memories": self.read_memories(customer_id),
            "issue_patterns": self.summarize_issue_patterns(customer_id),
        }
