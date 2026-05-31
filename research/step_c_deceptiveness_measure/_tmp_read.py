import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
print(open(sys.argv[1], encoding="utf-8").read())
