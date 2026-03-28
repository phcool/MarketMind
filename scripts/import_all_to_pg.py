"""
Deprecated: bulk import from local sector/stock folder trees.

The repository no longer ships import_forum_to_pg / import_eastmoney_news_to_pg /
import_sina_to_pg. Use the fetch scripts instead; they write directly to PostgreSQL:

  python scripts/fetch_forum_all.py
  python scripts/fetch_news_eastmoney.py --platform eastmoney
  python scripts/fetch_news_eastmoney.py --platform sina
"""

from __future__ import annotations

import sys


def main() -> None:
    print(__doc__, file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
