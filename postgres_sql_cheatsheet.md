# PostgreSQL & SQL Quick Reference

## Database Operations

### List Databases
```sql
-- psql command
\l

-- SQL query
SELECT datname FROM pg_database;
```

### Create Database
```sql
CREATE DATABASE my_app_db;
```

### Switch/Connect to Database
```sql
-- In psql
\c database_name

-- Example
\c my_app_db
```

### Delete Database
```sql
DROP DATABASE my_app_db;
```

## Table Operations

### List Tables
```sql
-- psql command
\dt

-- SQL query
SELECT tablename FROM pg_tables WHERE schemaname = 'public';
```

### Create Table
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Describe Table Structure
```sql
-- psql command
\d table_name

-- Example
\d users
```

### Delete Table
```sql
DROP TABLE users;
```

## Basic Data Operations

### Insert Data
```sql
-- Single row
INSERT INTO users (email, name) VALUES ('user@email.com', 'John');

-- Multiple rows
INSERT INTO users (email, name) VALUES 
    ('user1@email.com', 'Alice'),
    ('user2@email.com', 'Bob');
```

### Read Data
```sql
-- All data
SELECT * FROM users;

-- Specific columns
SELECT email, name FROM users;

-- With conditions
SELECT * FROM users WHERE name = 'John';
SELECT * FROM users WHERE created_at > '2024-01-01';

-- Limit results
SELECT * FROM users LIMIT 10;
```

### Update Data
```sql
UPDATE users SET name = 'John Doe' WHERE email = 'user@email.com';
```

### Delete Data
```sql
-- Delete specific rows
DELETE FROM users WHERE email = 'user@email.com';

-- Delete all data (keep table)
DELETE FROM users;
```

## Common Queries

### Count Records
```sql
SELECT COUNT(*) FROM users;
```

### Order Results
```sql
SELECT * FROM users ORDER BY created_at DESC;
SELECT * FROM users ORDER BY name ASC;
```

### Search Text
```sql
SELECT * FROM users WHERE name LIKE '%John%';
SELECT * FROM users WHERE email LIKE 'admin%';
```

## psql Commands

```sql
\l              -- List databases
\c db_name      -- Connect to database
\dt             -- List tables
\d table_name   -- Describe table
\q              -- Quit psql
\?              -- Help
\conninfo       -- Show connection info
```

## Environment Variables

```bash
export PGHOST=your-server.com
export PGPORT=5432
export PGUSER=username
export PGPASSWORD=password
export PGDATABASE=database_name
```

## Connection String
```bash
export DATABASE_URL="postgresql://user:password@host:port/database?sslmode=require"
```

## Essential Functions

```sql
-- Current time
SELECT NOW();

-- String functions
SELECT UPPER(name), LOWER(email) FROM users;

-- Count by group
SELECT is_active, COUNT(*) FROM users GROUP BY is_active;
```

## Indexes (Performance)

```sql
-- Create index
CREATE INDEX idx_users_email ON users(email);

-- List indexes
\di
```

## Backup & Restore

```bash
# Backup database
pg_dump database_name > backup.sql

# Restore database
psql database_name < backup.sql
```