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
- default ascending (升序) order, or noted with `ASC`.
- `BY` specifies one or more of the Column.
- `DESC` is the note of descending (降序).
```SQL
SELECT * FROM player ORDER BY level;
SELECT * FROM player ORDER BY 5;
SELECT * FROM player ORDER BY level DESC, exp ASC;
```
- The 3rd row is ordered by level descending first, and then ordered by exp ascending.
- If the level column is the 5th column, the 2nd row is equal to the first.

# Aggregate Functions
聚合函数用于对表中的数据进行计算和汇总，并返回一个单一的结果值。常见的表聚合函数包括：

COUNT：计算指定列或行的数量。
SUM：计算指定列的总和。
AVG：计算指定列的平均值。
MAX：找到指定列的最大值。
MIN：找到指定列的最小值。
GROUP BY：按指定列对结果进行分组。
HAVING：在GROUP BY之后，对分组结果进行筛选。
DISTINCT：返回唯一的列值。

```SQL
SELECT AVG(level) FROM player;

SELECT sex, COUNT(*) from player group by sex;
```

# HAVING
筛选出分组后的数据
```SQL
SELECT sex, COUNT(*) from player group by sex HAVING COUNT(level) > 4;
```
- only remains the count numbers of the players whose level > 4
- 