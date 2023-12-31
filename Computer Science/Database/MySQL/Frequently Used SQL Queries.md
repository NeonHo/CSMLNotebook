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

SELECT sex, COUNT(*) from player group by level order by count(level) DESC;
```
- First group by level
- then ordered by the count of the level.
# HAVING
筛选出分组后的数据
```SQL
SELECT sex, COUNT(*) from player group by sex HAVING COUNT(level) > 4;
```
- only remains the count numbers of the players whose level > 4
# SUBSTR()
```SQL
SELECT SUBSTR(name, 1, 1) from player
```
- the first parameter is column name
- the 2nd is the start position
- the 3rd is the length of substring.
# LIMIT
```SQL
SELECT SUBSTR(name, 1, 1), COUNT(SUBSTR(name, 1, 1)) from palyer
GROUP BY SUBSTR(name, 1, 1)
HAVING COUNT(SUBSTR(name, 1, 1)) >= 5
LIMIT 3
```
- `LIMIT` is used to remains the first 3 rows.
- `LIMIT 3, 3` is used to remains the 4~6 rows
	- the first 3 is the start index.
	- the 2nd is the length.
	- JD’s preference select.
# DISTINCT
keep rows unique.
Remove the repeating rows.
```SQL
SELECT DISTINCT sex from player
```
- The return table is `m, w` not `m, m, m, m…,w,…,w`.
# UNION
```SQL
SELECT * FROM player WHERE level BETWEEN 1 AND 3;
UNION
SELECT * FROM player WHERE exp BETWEEN 1 AND 3;
```
- Do $\cap$ for two tables.
- Remove the repeat record.
- `ALL` can remain repetition.
- Effective equal to `OR`
# INTERSECT
```SQL
SELECT * FROM player WHERE level BETWEEN 1 AND 3;
INTERSECT
SELECT * FROM player WHERE exp BETWEEN 1 AND 3;
```
$\cap$
# EXCEPT
```SQL
SELECT * FROM player WHERE level BETWEEN 1 AND 3;
EXCEPT
SELECT * FROM player WHERE exp BETWEEN 1 AND 3;
```
$A - B$

