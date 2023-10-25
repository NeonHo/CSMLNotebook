# WHERE
Use together with `SELECT, UPDATE, DELETE`.
```SQL
SELECT * FROM player WHERE level > 1 AND level < 5;
SELECT * FROM player WHERE level BETWEEN 1 AND 5;
SELECT * FROM player WHERE level NOT BETWEEN 1 AND 5;

SELECT * FROM player WHERE level > 1 AND level < 5 OR exp < 1 AND exp > 5;
SELECT * FROM player WHERE level IN (1, 3, 5);

SELECT * FROM player WHERE name LIKE `wang%`;
SELECT * FROM player WHERE name LIKE `%wang%`;

SELECT * FROM player WHERE name LIKE `wang_`;
SELECT * FROM player WHERE name REGEXP `^wang.$`;

SELECT * FROM player WHERE email is null;
SELECT * FROM player WHERE email = ‘’;
SELECT * FROM player WHERE email is NOT null;
```
- `OR`’s priority is lower than `AND`.
- `IN` is used for multiple selection.
- `NOT` means taking the opposite.
- `LIKE` is used to match strings.
- `REGEXP`
	- `.` Any single character
	- `^` head
	- `$` tail
	- `[abc]` Any single character in the list.
	- `[a-z]` Any single character in the range.
	- `A|B` ’A’ or ’B’.
- `null` uses `is` instead of `=`
- `null` and `’’` is not the same, which uses `=`
# ORDER BY
- default ascending (升序) order, or noted with `ASC`
- `BY` specifies one or more of the Column
- `DESC` is the note of descending (降序).
```SQL
SELECT * FROM player ORDER BY level;
SELECT * FROM player ORDER BY level , exp;
```