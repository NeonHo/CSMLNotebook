# WHERE
Use together with `SELECT, UPDATE, DELETE`.
```SQL
SELECT * FROM player WHERE level = 1 AND level < 5;
SELECT * FROM player WHERE level BETWEEN 1 AND 5;
SELECT * FROM player WHERE level NOT BETWEEN 1 AND 5;

SELECT * FROM player WHERE level > 1 AND level < 5 OR exp < 1 AND exp > 5;
SELECT * FROM player WHERE level IN (1, 3, 5);

SELECT * FROM player WHERE name LIKE `wang%`;
SELECT * FROM player WHERE name LIKE `%wang%`;
SELECT * FROM player WHERE name LIKE `wang__`;
SELECT * FROM player WHERE name REGEXP `wang.`;
```
- `OR`’s priority is lower than `AND`.
- `IN` is used for multiple selection.
- `NOT` means taking the opposite.
- `LIKE` is used to match strings.