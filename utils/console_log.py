import time


def info(message):
    print(f"[{time.strftime('%m-%d %H:%M:%S', time.localtime())}] [INFO] " + message.__str__())


def warn(message):
    print(f"[{time.strftime('%m-%d %H:%M:%S', time.localtime())}] [WARN] " + message.__str__())


def error(message):
    print(f"[{time.strftime('%m-%d %H:%M:%S', time.localtime())}] [ERROR] " + message.__str__())
