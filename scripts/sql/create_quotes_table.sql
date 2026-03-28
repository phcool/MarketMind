-- Daily OHLCV quotes: one row per (symbol, trade_date).
-- Run once (or rely on quotes_db.ensure_table / fetch_stocks first run).

CREATE TABLE IF NOT EXISTS quotes (
    symbol VARCHAR(10) NOT NULL,
    trade_date DATE NOT NULL,
    open NUMERIC(14, 4),
    close NUMERIC(14, 4),
    high NUMERIC(14, 4),
    low NUMERIC(14, 4),
    volume BIGINT,
    amount NUMERIC(22, 6),
    amplitude NUMERIC(10, 4),
    pct_change NUMERIC(10, 4),
    change_amount NUMERIC(14, 4),
    turnover NUMERIC(10, 4),
    PRIMARY KEY (symbol, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_quotes_symbol ON quotes (symbol);
CREATE INDEX IF NOT EXISTS idx_quotes_trade_date ON quotes (trade_date);
