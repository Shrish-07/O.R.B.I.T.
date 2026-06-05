from datetime import datetime
p='logs/actions.log'
with open(p,'a',encoding='utf8') as f:
    f.write(f"{datetime.utcnow().isoformat()}Z - INTERVALS WIRED: prediction intervals displayed; smoke test PASS on 8507\n")
print('Appended log')
