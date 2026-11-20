from datetime import datetime, timedelta, timezone

FMT = "%Y-%m-%dT%H:%M:%S.%f"


def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt=None):
    return (dt or utcnow()).strftime(FMT)


def parse(s):
    return datetime.strptime(s, FMT)


def ago(**kw):
    return iso(utcnow() - timedelta(**kw))


def later(**kw):
    return iso(utcnow() + timedelta(**kw))
