from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_URL = "mysql+pymysql://appuser:apppass@127.0.0.1:3306/customer_service"


def reset_demo_data() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
    engine = create_engine(database_url, future=True)

    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO customers (customer_id, name, email)
                VALUES
                  (1, 'Alice Chen', 'alice@example.com'),
                  (2, 'Bob Lin', 'bob@example.com'),
                  (3, 'Carol Wang', 'carol@example.com')
                ON DUPLICATE KEY UPDATE
                  name = VALUES(name),
                  email = VALUES(email)
                """
            )
        )
        connection.execute(
            text(
                """
                INSERT INTO orders (order_id, customer_id, product_name, status, order_date, delivery_date)
                VALUES
                  (12345, 1, 'Wireless Earbuds', 'in_transit', '2026-05-01 10:00:00', NULL),
                  (1001, 1, 'Smart Lamp', 'processing', '2026-05-02 12:00:00', NULL),
                  (3333, 1, 'Travel Mug', 'delivered', '2026-04-10 09:20:00', '2026-04-16 16:40:00'),
                  (5678, 2, 'Gaming Keyboard', 'delivered', '2026-04-20 09:00:00', '2026-04-24 15:00:00'),
                  (2222, 3, 'Laptop Stand', 'delivered', '2026-04-18 14:00:00', '2026-04-23 11:30:00'),
                  (7890, 1, 'USB-C Dock', 'delivered', '2026-04-19 08:30:00', '2026-04-22 13:15:00')
                ON DUPLICATE KEY UPDATE
                  customer_id = VALUES(customer_id),
                  product_name = VALUES(product_name),
                  status = VALUES(status),
                  order_date = VALUES(order_date),
                  delivery_date = VALUES(delivery_date)
                """
            )
        )
        connection.execute(text("DELETE FROM complaints"))
        connection.execute(
            text(
                """
                INSERT INTO complaints (customer_id, order_id, issue, status)
                VALUES
                  (1, 3333, 'late delivery', 'closed')
                """
            )
        )
        connection.execute(text("DELETE FROM customer_memory"))


def main() -> int:
    reset_demo_data()
    print(json.dumps({"status": "reset"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
