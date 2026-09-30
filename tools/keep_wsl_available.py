"""Keep a foreground WSL connection until the dispatch window and active jobs end."""
import json
import os
from pathlib import Path
import subprocess
import time
root=Path('/home/dhruv/projects/MalleableAI-FPGA/build/zephyrus-jobs/release/20260928T223413Z-b463e746/continuation-20260930')
control=json.loads((root/'control.json').read_text())
environment=dict(os.environ,XDG_RUNTIME_DIR='/run/user/1000',DBUS_SESSION_BUS_ADDRESS='unix:path=/run/user/1000/bus')
while True:
    units=subprocess.run(['systemctl','--user','list-units','--type=service','--state=running','--no-legend','--no-pager'],
        env=environment,capture_output=True,text=True)
    active='malleable-release-20260930' in units.stdout
    status={'status':'active','pid':os.getpid(),'checked_unix':time.time(),
        'dispatch_deadline_unix':control['dispatch_deadline_unix'],'release_job_active':active}
    (root/'wsl-keeper-status.json').write_text(json.dumps(status))
    if time.time()>=control['dispatch_deadline_unix'] and not active and units.returncode==0:break
    time.sleep(15)
(root/'wsl-keeper-status.json').write_text(json.dumps(dict(status,status='released')))
