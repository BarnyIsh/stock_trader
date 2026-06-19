"""
scheduler.py - Run the trading bot on a market-hours schedule

Usage:
  python scheduler.py               # runs every weekday at market open (9:31 AM ET)
  python scheduler.py --now         # run immediately (for testing)
  python scheduler.py --cron        # print crontab line and exit

For production, prefer cron:
  31 9 * * 1-5 cd /path/to/algo_trader && python trader.py --mode run >> logs/cron.log 2>&1
"""

import time
import argparse
import subprocess
from datetime import datetime, timezone
import pytz

MARKET_TZ  = pytz.timezone("America/New_York")
RUN_HOUR   = 9
RUN_MINUTE = 31   # 9:31 AM ET — 1 min after open so prices stabilise

MARKET_DAYS = {0, 1, 2, 3, 4}   # Mon–Fri


def is_market_day() -> bool:
    return datetime.now(MARKET_TZ).weekday() in MARKET_DAYS


def seconds_until_next_run() -> float:
    now = datetime.now(MARKET_TZ)
    target = now.replace(hour=RUN_HOUR, minute=RUN_MINUTE, second=0, microsecond=0)
    if now >= target:
        # Already past today's run time — schedule for next weekday
        from datetime import timedelta
        target += timedelta(days=1)
        while target.weekday() not in MARKET_DAYS:
            target += timedelta(days=1)
    return (target - now).total_seconds()


def run_bot():
    print(f"\n🚀 Triggering trading session at {datetime.now(MARKET_TZ).strftime('%Y-%m-%d %H:%M %Z')}")
    result = subprocess.run(
        ["python", "trader.py", "--mode", "run"],
        capture_output=False,
    )
    if result.returncode != 0:
        print("[error] trader.py exited with non-zero code.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--now",  action="store_true", help="Run immediately")
    parser.add_argument("--cron", action="store_true", help="Print crontab entry")
    args = parser.parse_args()

    if args.cron:
        print("# Add to crontab with: crontab -e")
        print(f"31 9 * * 1-5 cd $(pwd) && python trader.py --mode run >> logs/cron.log 2>&1")
        return

    if args.now:
        run_bot()
        return

    print("⏰ Algo Trader Scheduler started.")
    print(f"   Will run at {RUN_HOUR:02d}:{RUN_MINUTE:02d} ET on weekdays.\n")

    while True:
        wait = seconds_until_next_run()
        h, m = divmod(int(wait), 3600)
        m, s = divmod(m, 60)
        print(f"   Next run in {h}h {m}m {s}s …", end="\r", flush=True)
        time.sleep(min(wait, 60))   # check every minute
        if wait <= 60 and is_market_day():
            run_bot()
            time.sleep(120)   # prevent double-run


if __name__ == "__main__":
    main()
