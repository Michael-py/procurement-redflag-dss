#!/bin/sh
# Runs once, when the data volume is first created. Creates a login that can
# only read: the dashboard connects with it, so it cannot change the warehouse.
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v reader_password="$READER_PASSWORD" <<'SQL'
CREATE ROLE redflag_reader LOGIN PASSWORD :'reader_password';
GRANT CONNECT ON DATABASE :"DBNAME" TO redflag_reader;
SQL
