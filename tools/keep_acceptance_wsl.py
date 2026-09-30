"""Hold WSL open for the finite acceptance unit; never dispatch model jobs."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);unit=sys.argv[2];start=time.time();seen=False
env=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
while True:
    query=subprocess.run(['systemctl','--user','show',unit,'-p','ActiveState','--value'],env=env,capture_output=True,text=True,timeout=15)
    state=query.stdout.strip();active=state in ('active','activating','deactivating');seen|=active
    status={'pid':os.getpid(),'unit':unit,'state':state,'checked_unix':time.time(),'status':'holding-wsl'}
    (root/'keeper-status.json').write_text(json.dumps(status))
    if not active and (seen or (root/'terminal.json').exists() or time.time()-start>300):break
    time.sleep(15)
(root/'keeper-status.json').write_text(json.dumps(dict(status,status='released')))
