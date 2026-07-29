import os
import sys
import time
import subprocess
from apscheduler.triggers.cron import CronTrigger
from apscheduler.schedulers.background import BackgroundScheduler

PID_FILE = "scheduler.pid"
scheduler = BackgroundScheduler()

def start_server():
    subprocess.run(
        ["/bin/bash", "/home/workspace/medflow/scripts/start-server.sh", "start"]
    )

def start_job():
    scheduler.add_job(start_server, CronTrigger.from_crontab('0 0 * * *'), id="every_day")
    scheduler.start()
    print(f"\nJob has been started.")

def stop_job():
    scheduler.shutdown(wait=False)
    print(f"\nJob has been stopped.")

def read_pid():
    with open(PID_FILE, "r") as f:
        pid = int(f.read())
    return pid

def write_pid():
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

def remove_pid():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("\nUsage：python scheduler.py start|stop")
        sys.exit(1)

    command = sys.argv[1]
    match command:
        case "start":
            if os.path.exists(PID_FILE):
                print("\nThe service has been started. Please execute the command \"python scheduler.py stop\" to stop the service.")
                sys.exit(1)
            write_pid()
            start_job()
            try:
                while True:
                    time.sleep(10)
            except KeyboardInterrupt:
                stop_job()
                remove_pid()
        case "stop":
            if not os.path.exists(PID_FILE):
                print("\nThe service has not been started. Please execute the command \"python scheduler.py start\" to start the service.")
                sys.exit(1)
            pid = read_pid()
            remove_pid()
            try:
                os.kill(pid, 9)
                print(f"\nKill PID: {pid}")
            except ProcessLookupError:
                print(f"\nUnknown PID: {pid}")
            #stop_job()
        case _:
            print(f"\nUnknown parameters: {command}, only support start or stop.")
            