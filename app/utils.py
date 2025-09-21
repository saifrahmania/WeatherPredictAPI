from datetime import date, timedelta

def plus_days(d: date, n: int) -> date:
    return d + timedelta(days=n)
