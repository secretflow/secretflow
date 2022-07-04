import platform


def rows_count(filename):
    """get rows count from file"""
    line_break = b'\r\n' if platform.system().lower() == "windows" else b'\n'
    with open(filename, "rb") as f:
        count = 0
        buf_size = 1024 * 1024
        buf = f.read(buf_size)
        while buf:
            count += buf.count(line_break)
            buf = f.read(buf_size)
        return count
