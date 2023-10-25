# WHERE
Use together with `SELECT, UPDATE, DELETE`.
```SQL
SELECT * FROM player WHERE level = 1 AND level < 5;
SELECT * FROM player WHERE level > 1 AND level < 5 OR exp < 1 AND exp > 5;
```