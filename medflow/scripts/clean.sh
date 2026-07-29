#!/bin/bash

log_file=$1
openai_api_server_port=$2
inference_server_port=$3
web_server_port=$4
case2chat_port=$5

rm -rf ./${log_file} ../src/*.json ../src/*.xlsx
#rm -rf nohup.out

echo "====== Clean ======" >> $log_file
pid=$(ps -aux | grep vllm | grep ${openai_api_server_port} | grep -v 'grep ' | awk '{print $2}')
if [ -n "$pid" ]; then
  pkill -P $pid
  kill -9 $pid
fi

kill -9 $(lsof -i :${inference_server_port} -t)

kill -9 $(ps -aux | grep inference_ui | grep ${web_server_port} | grep -v 'grep ' | awk '{print $2}')

kill -9 $(lsof -i :${case2chat_port} -t)
echo "====== Clean End ======" >> $log_file
