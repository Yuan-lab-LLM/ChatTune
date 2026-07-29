#! /bin/bash

excel_path=../data/raw/excel
database_path=../data/processed/database
fastbm25_path=../data/processed/fastbm25

log_file=create-database.log

rm -rf ./${log_file}
#rm -rf ./nohup.out

echo "python3 ../data/create_database.py --excel-path "  ${excel_path} "--database-path" ${database_path} "--fastbm25-path" ${fastbm25_path} >> $log_file
nohup python3 ../data/create_database.py --excel-path  ${excel_path} --database-path ${database_path} --fastbm25-path ${fastbm25_path}

echo "##################### load data successfully." >> $log_file