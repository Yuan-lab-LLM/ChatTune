#! /bin/bash

# The inference service is restarted at 00:00 every day

log_file=scheduler.log
pid_file=scheduler.pid

if [ -z "$1" ]; then
    echo "Usage：bash scheduler.sh start|stop" >> $log_file
    exit 1
fi

if [ "$1" == "start" ]; then
    if [ -f "$pid_file" ]; then
        echo "The service has been started. Please execute the command \"bash scheduler.sh stop\" to stop the service." >> $log_file
        exit 1
    fi
    echo "nohup python3 ../src/common/scheduler.py start &" >> $log_file
    nohup python3 ../src/common/scheduler.py start &
    sleep 1
elif [ "$1" == "stop" ]; then
    echo "nohup python3 ../src/common/scheduler.py stop &" >> $log_file
    nohup python3 ../src/common/scheduler.py stop &
    sleep 1
    rm -rf ./$log_file ./$pid_file
#    rm -rf ./nohup.out
else
    echo "Unknown parameters: $1, only support start or stop." >> $log_file
fi
